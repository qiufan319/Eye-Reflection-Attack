"""
Composite loss function and FOMAML meta-learning for
conditional diffusion-based multi-frame super-resolution.
"""

import random
from collections import defaultdict
from math import exp
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from guided_diffusion.gaussian_diffusion import ModelMeanType

# ---------------------------------------------------------------------------
# Re-use existing MS-SSIM implementation
# ---------------------------------------------------------------------------
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
from mssim import MSSSIM


# ===================================================================
#  Section 1: Utility helpers
# ===================================================================


def predict_x0(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
    diffusion,
) -> torch.Tensor:
    """
    Recover the clean-image estimate x_hat_0 from a single denoising
    step, handling all ModelMeanType variants and optional learned
    variance.

    Args:
        x_t:          noisy image at timestep t   [B, C, H, W]
        t:            timestep indices             [B]
        model_output: raw network output           [B, C' , H, W]
        diffusion:    GaussianDiffusion(Bsr) instance

    Returns:
        x0_pred clamped to [-1, 1]                 [B, C, H, W]
    """
    C = x_t.shape[1]
    # If the model predicts both mean and variance, keep only the mean half.
    if model_output.shape[1] == C * 2:
        model_output, _ = torch.split(model_output, C, dim=1)

    if diffusion.model_mean_type == ModelMeanType.START_X:
        x0_pred = model_output
    elif diffusion.model_mean_type == ModelMeanType.EPSILON:
        x0_pred = diffusion._predict_xstart_from_eps(x_t, t, model_output)
    elif diffusion.model_mean_type == ModelMeanType.PREVIOUS_X:
        x0_pred = diffusion._predict_xstart_from_xprev(x_t, t, model_output)
    else:
        raise NotImplementedError(diffusion.model_mean_type)

    return x0_pred.clamp(-1.0, 1.0)


def _make_gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Create a 2-D Gaussian blur kernel for depthwise convolution."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)  # outer product
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    return kernel_2d.repeat(channels, 1, 1, 1)  # [C, 1, K, K]


def _make_sobel_kernels(device: torch.device, dtype: torch.dtype):
    """Sobel gradient filters for 3-channel images (depthwise)."""
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return gx, gy


# ===================================================================
#  Section 2: VGG-19 multi-layer feature extractor
# ===================================================================


class VGGFeatureExtractor(nn.Module):
    """
    Standalone VGG-19 multi-layer feature extractor for 3-channel RGB
    images in [-1, 1] range.  Internally converts to [0, 1] and applies
    ImageNet normalisation.  All VGG parameters are frozen; gradient
    flows only through the *input* tensor.
    """

    # VGG-19 feature indices (0-based) right after each named ReLU
    LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_4": 18,
        "relu4_4": 27,
    }

    def __init__(
        self,
        layers: Tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
    ):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        max_idx = max(self.LAYER_MAP[l] for l in layers)
        # Store only the layers we need
        self.slices = nn.Sequential(*list(vgg.children())[:max_idx])
        self.target_indices = {self.LAYER_MAP[l]: l for l in layers}

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        # Freeze all VGG parameters
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # [-1, 1] -> [0, 1] -> ImageNet-normalised
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std

        features: Dict[str, torch.Tensor] = {}
        for idx, module in enumerate(self.slices):
            x = module(x)
            real_idx = idx + 1  # Sequential is 0-indexed
            if real_idx in self.target_indices:
                features[self.target_indices[real_idx]] = x
        return features


# ===================================================================
#  Section 3: Composite loss
# ===================================================================


class CompositeSRLoss(nn.Module):
    """
    Five-term composite training loss for conditional diffusion-based
    multi-frame multi-frame super-resolution.

    Terms
    -----
    L_diff    : diffusion noise-prediction MSE (passed in, not recomputed)
    L_color   : Gaussian-blur colour / distribution consistency (L1)
    L_perc    : VGG-19 multi-layer perceptual loss (L1)
    L_tex     : Sobel-gradient Charbonnier edge loss
    L_ssim    : 1 - MS-SSIM

    All image tensors are expected in [-1, 1] range.
    """

    def __init__(
        self,
        device: torch.device,
        w_diffusion: float = 1.0,
        w_color: float = 0.4,
        w_perceptual: float = 0.2,
        w_texture: float = 0.1,
        w_ssim: float = 0.1,
        gaussian_kernel_size: int = 11,
        gaussian_sigma: float = 1.5,
        vgg_layers: Tuple[str, ...] = ("relu1_2", "relu2_2", "relu3_4", "relu4_4"),
        charbonnier_eps: float = 1e-3,
    ):
        super().__init__()
        self.w_diff = w_diffusion
        self.w_color = w_color
        self.w_perc = w_perceptual
        self.w_tex = w_texture
        self.w_ssim = w_ssim
        self.charbonnier_eps = charbonnier_eps
        self.gauss_pad = gaussian_kernel_size // 2

        # Gaussian blur kernel (depthwise, 3-channel)
        gauss_k = _make_gaussian_kernel(gaussian_kernel_size, gaussian_sigma, 3)
        self.register_buffer("gaussian_kernel", gauss_k)

        # Sobel kernels
        gx, gy = _make_sobel_kernels(device, torch.float32)
        self.register_buffer("sobel_gx", gx)
        self.register_buffer("sobel_gy", gy)

        # VGG feature extractor (frozen)
        self.vgg = VGGFeatureExtractor(layers=vgg_layers).to(device).eval()

        # MS-SSIM module (from model/mssim.py)
        self.msssim = MSSSIM(window_size=11, size_average=True, channel=3)

        # Move all buffers/submodules to device
        self.to(device)

    # ---- individual loss terms -----------------------------------------

    def _color_loss(self, x0_pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Gaussian-blur both images, then L1.  Alignment-tolerant."""
        x_blur = F.conv2d(x0_pred, self.gaussian_kernel, padding=self.gauss_pad, groups=3)
        g_blur = F.conv2d(gt, self.gaussian_kernel, padding=self.gauss_pad, groups=3)
        return F.l1_loss(x_blur, g_blur)

    def _perceptual_loss(self, x0_pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Multi-layer VGG-19 feature L1 distance."""
        pred_feats = self.vgg(x0_pred)
        with torch.no_grad():
            gt_feats = self.vgg(gt)
        total = torch.tensor(0.0, device=x0_pred.device)
        for key in pred_feats:
            total = total + F.l1_loss(pred_feats[key], gt_feats[key])
        return total / len(pred_feats)

    def _texture_loss(self, x0_pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Sobel-gradient Charbonnier edge loss."""
        eps2 = self.charbonnier_eps ** 2
        dx_pred = F.conv2d(x0_pred, self.sobel_gx, padding=1, groups=3)
        dx_gt = F.conv2d(gt, self.sobel_gx, padding=1, groups=3)
        dy_pred = F.conv2d(x0_pred, self.sobel_gy, padding=1, groups=3)
        dy_gt = F.conv2d(gt, self.sobel_gy, padding=1, groups=3)
        loss_x = torch.mean(torch.sqrt((dx_pred - dx_gt) ** 2 + eps2))
        loss_y = torch.mean(torch.sqrt((dy_pred - dy_gt) ** 2 + eps2))
        return loss_x + loss_y

    def _ssim_loss(self, x0_pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """1 - MS-SSIM (auto-detects val_range for [-1,1] images)."""
        return 1.0 - self.msssim(x0_pred, gt)

    # ---- main forward --------------------------------------------------

    def forward(
        self,
        x0_pred: torch.Tensor,
        gt: torch.Tensor,
        diffusion_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            x0_pred:        predicted clean image  [B, 3, H, W]  in [-1, 1]
            gt:             ground truth           [B, 3, H, W]  in [-1, 1]
            diffusion_loss: scalar, pre-computed noise-prediction MSE

        Returns:
            total_loss:  scalar tensor (with grad)
            loss_dict:   dict of detached float values for logging
        """
        L_color = self._color_loss(x0_pred, gt)
        L_perc = self._perceptual_loss(x0_pred, gt)
        L_tex = self._texture_loss(x0_pred, gt)
        L_ssim = self._ssim_loss(x0_pred, gt)

        total = (
            self.w_diff * diffusion_loss
            + self.w_color * L_color
            + self.w_perc * L_perc
            + self.w_tex * L_tex
            + self.w_ssim * L_ssim
        )

        loss_dict = {
            "diffusion": diffusion_loss.item(),
            "color": L_color.item(),
            "perceptual": L_perc.item(),
            "texture": L_tex.item(),
            "ssim": L_ssim.item(),
            "total": total.item(),
        }
        return total, loss_dict


# ===================================================================
#  Section 4: Cross-subject FOMAML meta-learning
# ===================================================================


def split_batch_by_subject(
    subject_ids: torch.Tensor,
) -> Dict[int, List[int]]:
    """Group batch indices by subject identity.

    Args:
        subject_ids: [B] tensor of integer subject labels.

    Returns:
        {subject_id: [batch_index, ...]}
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, sid in enumerate(subject_ids.tolist()):
        groups[sid].append(idx)
    return dict(groups)


def sample_meta_episode(
    subjects: Dict[int, List[int]],
    source_ratio: float = 0.7,
) -> Tuple[List[int], List[int]]:
    """Randomly split subjects into source and target sets.

    Returns:
        (source_batch_indices, target_batch_indices)
    """
    keys = list(subjects.keys())
    random.shuffle(keys)
    n_source = max(1, int(len(keys) * source_ratio))
    # Ensure at least 1 target subject
    if n_source >= len(keys):
        n_source = len(keys) - 1
    # Degenerate case: single subject -> use same data for both
    if len(keys) == 1:
        return subjects[keys[0]], subjects[keys[0]]

    source_keys = keys[:n_source]
    target_keys = keys[n_source:]
    source_idx = [i for k in source_keys for i in subjects[k]]
    target_idx = [i for k in target_keys for i in subjects[k]]
    return source_idx, target_idx


def _forward_and_loss(
    model,
    frame_encoder,
    implicit_aligner,
    burst: torch.Tensor,
    extra_burst: torch.Tensor,
    gt: torch.Tensor,
    loss_fn: CompositeSRLoss,
    diffusion,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Shared forward pass: feature extraction -> diffusion loss -> composite loss."""
    b, n, c, h, w = extra_burst.shape
    feats = frame_encoder(extra_burst.view(b * n, c, h, w)).view(b, n, -1, h, w)
    _, fused = implicit_aligner(feats)
    extra_features_by_t = fused.unsqueeze(0).expand(
        diffusion.num_timesteps, -1, -1, -1, -1
    )

    t = torch.randint(0, diffusion.num_timesteps, (gt.shape[0],), device=device)
    losses = diffusion.training_losses(
        model, gt, t,
        model_kwargs={"y": burst, "extra_features_by_t": extra_features_by_t},
    )
    diffusion_loss = losses["loss"].mean()

    # Reconstruct x0_pred for auxiliary losses
    noise = torch.randn_like(gt)
    x_t = diffusion.q_sample(x_start=gt, t=t, noise=noise)
    model_out = model(x_t, t, y=burst, extra_features_by_t=extra_features_by_t)
    x0_pred = predict_x0(x_t, t, model_out, diffusion)

    total_loss, loss_dict = loss_fn(x0_pred, gt, diffusion_loss)
    return total_loss, loss_dict


def fomaml_inner_loop(
    model: nn.Module,
    frame_encoder: nn.Module,
    implicit_aligner: nn.Module,
    source_data: Dict[str, torch.Tensor],
    loss_fn: CompositeSRLoss,
    diffusion,
    inner_lr: float = 1e-5,
    inner_steps: int = 1,
    device: torch.device = torch.device("cuda"),
) -> None:
    """
    FOMAML inner loop: perform `inner_steps` SGD updates on source-
    subject data *in place*.  The caller is responsible for saving and
    restoring the original parameters.
    """
    all_params = (
        list(model.parameters())
        + list(frame_encoder.parameters())
        + list(implicit_aligner.parameters())
    )

    for _ in range(inner_steps):
        total_loss, _ = _forward_and_loss(
            model, frame_encoder, implicit_aligner,
            source_data["burst"], source_data["extra_burst"], source_data["gt"],
            loss_fn, diffusion, device,
        )
        # Manual SGD using autograd.grad (does not touch .grad attributes)
        grads = torch.autograd.grad(total_loss, all_params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(all_params, grads):
                if g is not None:
                    p.sub_(inner_lr * g)


def meta_train_step(
    model: nn.Module,
    frame_encoder: nn.Module,
    implicit_aligner: nn.Module,
    batch_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    subject_ids: torch.Tensor,
    loss_fn: CompositeSRLoss,
    diffusion,
    inner_lr: float = 1e-5,
    inner_steps: int = 1,
    source_ratio: float = 0.7,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    One complete FOMAML meta-training step.

    1. Partition the batch by subject into source / target.
    2. Save original parameters.
    3. Inner-loop: SGD on source data (in-place).
    4. Compute loss on target data with adapted parameters.
    5. Backprop to get FOMAML meta-gradients on adapted params.
    6. Restore original parameters and attach the meta-gradients.

    The caller should then call ``optimizer.step()`` to apply the update.

    Returns:
        (meta_loss_detached, loss_dict)
    """
    burst, extra_burst, gt = batch_data

    # 1. Source / target split
    subjects = split_batch_by_subject(subject_ids)
    source_idx, target_idx = sample_meta_episode(subjects, source_ratio)

    source_data = {
        "burst": burst[source_idx],
        "extra_burst": extra_burst[source_idx],
        "gt": gt[source_idx],
    }
    target_data = {
        "burst": burst[target_idx],
        "extra_burst": extra_burst[target_idx],
        "gt": gt[target_idx],
    }

    # 2. Snapshot original parameters
    orig_states = {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "frame_encoder": {k: v.clone() for k, v in frame_encoder.state_dict().items()},
        "implicit_aligner": {k: v.clone() for k, v in implicit_aligner.state_dict().items()},
    }

    # 3. Inner loop on source subjects
    fomaml_inner_loop(
        model, frame_encoder, implicit_aligner,
        source_data, loss_fn, diffusion, inner_lr, inner_steps, device,
    )

    # 4. Target loss with adapted parameters
    meta_loss, loss_dict = _forward_and_loss(
        model, frame_encoder, implicit_aligner,
        target_data["burst"], target_data["extra_burst"], target_data["gt"],
        loss_fn, diffusion, device,
    )

    # 5. Backward on adapted parameters
    meta_loss.backward()

    # Save the FOMAML meta-gradients
    all_named = (
        list(model.named_parameters())
        + list(frame_encoder.named_parameters())
        + list(implicit_aligner.named_parameters())
    )
    meta_grads = {}
    for name, p in all_named:
        if p.grad is not None:
            meta_grads[name] = p.grad.clone()

    # 6. Restore original parameters
    model.load_state_dict(orig_states["model"])
    frame_encoder.load_state_dict(orig_states["frame_encoder"])
    implicit_aligner.load_state_dict(orig_states["implicit_aligner"])

    # Re-attach saved meta-gradients to the restored parameters
    for name, p in all_named:
        if name in meta_grads:
            p.grad = meta_grads[name]
        else:
            p.grad = None

    return meta_loss.detach(), loss_dict


def train_meta_learning(
    model: nn.Module,
    frame_encoder: nn.Module,
    implicit_aligner: nn.Module,
    diffusion,
    loss_fn: CompositeSRLoss,
    optimizer: torch.optim.Optimizer,
    loader,
    val_loader=None,
    epochs: int = 10000,
    device: torch.device = torch.device("cuda"),
    writer=None,
    inner_lr: float = 1e-5,
    inner_steps: int = 1,
    source_ratio: float = 0.7,
    meta_prob: float = 0.5,
    start_epoch: int = 1,
    global_step: int = 0,
    save_every: int = 100,
    save_path: str = "checkpoint.pt",
):
    """
    Outer meta-training loop.  With probability ``meta_prob`` each batch
    uses FOMAML; otherwise standard (non-meta) training is used for
    stability.

    Args:
        meta_prob: probability of using meta-learning per batch
                   (0.0 = pure standard training, 1.0 = always meta)
    """
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, (burst, extra_burst, gt, subject_ids) in enumerate(loader):
            burst = burst.to(device)
            extra_burst = extra_burst.to(device)
            gt = gt.to(device)
            subject_ids = subject_ids.to(device)

            use_meta = random.random() < meta_prob

            if use_meta:
                # --- FOMAML episode ---
                loss_val, loss_dict = meta_train_step(
                    model, frame_encoder, implicit_aligner,
                    (burst, extra_burst, gt), subject_ids,
                    loss_fn, diffusion,
                    inner_lr=inner_lr, inner_steps=inner_steps,
                    source_ratio=source_ratio, device=device,
                )
                optimizer.step()
                optimizer.zero_grad()
            else:
                # --- Standard training ---
                total_loss, loss_dict = _forward_and_loss(
                    model, frame_encoder, implicit_aligner,
                    burst, extra_burst, gt,
                    loss_fn, diffusion, device,
                )
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                loss_val = total_loss.detach()

            # Logging
            if writer is not None:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, global_step)
                writer.add_scalar("train/is_meta", float(use_meta), global_step)

            epoch_loss += loss_val.item()
            epoch_steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"epoch={epoch} avg_loss={avg_loss:.4f} steps={epoch_steps}")

        if epoch % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "frame_encoder": frame_encoder.state_dict(),
                "implicit_aligner": implicit_aligner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            }
            torch.save(ckpt, save_path)
            print(f"checkpoint saved: {save_path} (epoch {epoch})")

    if writer is not None:
        writer.close()


# ===================================================================
#  Section 5: Example usage
# ===================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Placeholder tensors for demonstration ---
    B, N, C, H, W = 4, 20, 3, 160, 160
    selected_frames = torch.randn(B, N, C, H, W, device=device)
    gt_hr = torch.randn(B, C, H, W, device=device).clamp(-1, 1)
    subject_ids = torch.tensor([0, 0, 1, 1], device=device)

    # --- Instantiate the composite loss ---
    loss_fn = CompositeSRLoss(device=device)

    # --- Example: compute loss given pre-computed quantities ---
    x0_pred = torch.randn(B, C, H, W, device=device, requires_grad=True).clamp(-1, 1)
    diffusion_loss = torch.tensor(0.05, device=device, requires_grad=True)

    total_loss, loss_dict = loss_fn(x0_pred, gt_hr, diffusion_loss)
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.6f}")
    print(f"Total loss (grad_fn={total_loss.grad_fn is not None}): {total_loss.item():.6f}")

    # --- Example: meta-learning split ---
    subjects = split_batch_by_subject(subject_ids)
    src, tgt = sample_meta_episode(subjects, source_ratio=0.5)
    print(f"\nMeta episode: source indices={src}, target indices={tgt}")
