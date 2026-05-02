import random
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.script_util import create_model_and_diffusion_bsr
from composite_loss import (
    CompositeSRLoss,
    predict_x0,
    meta_train_step,
    train_meta_learning,
)


def _dynamic_create_condition(self, y, extra_features=None):
    """
    Build condition dict for arbitrary image_size (e.g., 160x160).
    """
    y = self.make_condition_feature(y)
    if extra_features is not None:
        if self.extra_feature_projector is None:
            raise ValueError(
                "extra_features provided but extra_feature_channels/extra_cond_channels are not set"
            )
        if extra_features.shape[-2:] != y.shape[-2:]:
            extra_features = F.interpolate(
                extra_features, size=y.shape[-2:], mode="bicubic"
            )
        extra_features = self.extra_feature_projector(extra_features)
        y = torch.cat([y, extra_features], dim=1)
    elif self.extra_cond_channels > 0:
        zeros = torch.zeros(
            y.shape[0],
            self.extra_cond_channels,
            y.shape[2],
            y.shape[3],
            device=y.device,
            dtype=y.dtype,
        )
        y = torch.cat([y, zeros], dim=1)

    sizes = [self.image_size // (2**i) for i in range(len(self.channel_mult))]
    sizes = sorted({int(s) for s in sizes if s > 0})
    y_dic = {}
    for s in sizes:
        if s == y.shape[-1]:
            y_dic[str(s)] = y
        else:
            y_dic[str(s)] = F.interpolate(y, size=(s, s), mode="bicubic")
    return y_dic


def charbonnier_loss(x, y, eps=1e-3):
    return torch.mean(torch.sqrt((x - y) ** 2 + eps**2))


def make_sobel_kernels(device, dtype):
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    gx = gx.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gy = gy.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return gx, gy


def sobel_edge_loss(x, y, gx, gy):
    grad_x = F.conv2d(x, gx, padding=1, groups=3)
    grad_y = F.conv2d(y, gx, padding=1, groups=3)
    grad_x_y = F.conv2d(x, gy, padding=1, groups=3)
    grad_y_y = F.conv2d(y, gy, padding=1, groups=3)
    return torch.mean(torch.abs(grad_x - grad_y) + torch.abs(grad_x_y - grad_y_y))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(r"data")
    image_size = 160
    batch_size = 4
    burst_size = 20
    extra_burst_size = 20
    diffusion_steps = 500

    model, diffusion = create_model_and_diffusion_bsr(
        image_size=image_size,
        learn_sigma=False,
        num_channels=64,
        num_res_blocks=1,
        channel_mult="1,2,2,4",
        num_heads=2,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="40,20",
        dropout=0.0,
        diffusion_steps=diffusion_steps,
        noise_schedule="sigmoid",
        timestep_respacing="ddim50",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        burst_size=burst_size,
        num_cond_features=8,
        extra_feature_channels=32,
        extra_cond_channels=16,
    )
    model.to(device)

    # Patch create_condition to support image_size=160.
    model.create_condition = types.MethodType(_dynamic_create_condition, model)

    class SimpleConditionFeature(nn.Module):
        def __init__(self, num_features):
            super().__init__()
            self.conv = nn.Conv2d(3, num_features, kernel_size=3, padding=1)

        def forward(self, burst):
            b, n, c, h, w = burst.shape
            x = burst.view(b * n, c, h, w)
            x = self.conv(x)
            x = x.view(b, n * x.shape[1], h, w)
            return x

    model.make_condition_feature = SimpleConditionFeature(num_features=8).to(device)

    class FrameEncoder(nn.Module):
        def __init__(self, channels=32):
            super().__init__()
            self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.act = nn.SiLU()

        def forward(self, x):
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            return x

    class ImplicitAligner(nn.Module):
        def __init__(
            self, channels=32, burst_size=5, window_size=7, fused_channels=32, downsample_scale=1
        ):
            super().__init__()
            self.channels = channels
            self.burst_size = burst_size
            self.window_size = window_size
            self.padding = window_size // 2
            self.downsample_scale = downsample_scale
            self.fuse = nn.Conv2d(burst_size * channels, fused_channels, kernel_size=1)

        def _align_one(self, frame, ref):
            if self.downsample_scale > 1:
                h, w = frame.shape[-2:]
                frame = F.interpolate(
                    frame, size=(h // self.downsample_scale, w // self.downsample_scale), mode="bicubic", align_corners=False
                )
                ref = F.interpolate(
                    ref, size=(h // self.downsample_scale, w // self.downsample_scale), mode="bicubic", align_corners=False
                )
            b, c, h, w = frame.shape
            patches = F.unfold(frame, kernel_size=self.window_size, padding=self.padding)
            patches = patches.view(b, c, self.window_size * self.window_size, h * w)
            ref_center = ref.view(b, c, h * w).unsqueeze(2)
            attn = (patches * ref_center).sum(dim=1)
            attn = F.softmax(attn, dim=1)
            aligned = (patches * attn.unsqueeze(1)).sum(dim=2)
            aligned = aligned.view(b, c, h, w)
            if self.downsample_scale > 1:
                aligned = F.interpolate(
                    aligned, size=(frame.shape[-2] * self.downsample_scale, frame.shape[-1] * self.downsample_scale), mode="bicubic", align_corners=False
                )
            return aligned

        def forward(self, feats):
            b, n, c, h, w = feats.shape
            ref = feats[:, n // 2]
            aligned_list = []
            for i in range(n):
                aligned_list.append(self._align_one(feats[:, i], ref))
            aligned = torch.stack(aligned_list, dim=1)
            fused = self.fuse(aligned.view(b, n * c, h, w))
            return aligned, fused

    frame_encoder = FrameEncoder(channels=32).to(device)
    implicit_aligner = ImplicitAligner(
        channels=32, burst_size=extra_burst_size, window_size=7, fused_channels=32, downsample_scale=1
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(frame_encoder.parameters())
        + list(implicit_aligner.parameters()),
        lr=1e-4,
    )
    resume_path = ""  # set to checkpoint path to resume, e.g. r"epoch100.pt"
    start_epoch = 1
    global_step = 0
    if resume_path is not None and str(resume_path).strip() and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "frame_encoder" in ckpt:
            frame_encoder.load_state_dict(ckpt["frame_encoder"])
        if "implicit_aligner" in ckpt:
            implicit_aligner.load_state_dict(ckpt["implicit_aligner"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError as exc:
                print(f"optimizer state not loaded: {exc}")
                print("continuing with freshly initialized optimizer")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"resumed from {resume_path} at epoch {start_epoch}")

    class OpticalDataset(Dataset):
        def __init__(self, root, split="train", size=160, burst_size=1, extra_burst_size=1):
            self.lr_root = Path(root) / split / "LR_aligned"
            self.size = size
            self.burst_size = burst_size
            self.extra_burst_size = extra_burst_size
            self.folders = sorted([p for p in self.lr_root.iterdir() if p.is_dir()])

        def __len__(self):
            return len(self.folders)

        def _load_rgb(self, path):
            img = Image.open(path).convert("RGB")
            x = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(x)
            x = x.permute(2, 0, 1).contiguous()
            return x

        def __getitem__(self, idx):
            folder = self.folders[idx]
            files = sorted(folder.glob("*.png"))
            if len(files) < 6:
                raise ValueError(f"expected at least 6 images in {folder}, got {len(files)}")
            lr_files = files[:-1]
            if len(lr_files) < self.burst_size:
                raise ValueError(
                    f"expected at least {self.burst_size} LR images in {folder}, got {len(lr_files)}"
                )
            if len(lr_files) < self.extra_burst_size:
                raise ValueError(
                    f"expected at least {self.extra_burst_size} LR images in {folder}, got {len(lr_files)}"
                )
            if self.burst_size == 1:
                burst_files = [lr_files[0]]
            else:
                burst_files = lr_files[: self.burst_size]
            extra_burst_files = lr_files[: self.extra_burst_size]
            burst = [self._load_rgb(p) for p in burst_files]
            extra_burst = [self._load_rgb(p) for p in extra_burst_files]
            burst = torch.stack(burst, dim=0)
            extra_burst = torch.stack(extra_burst, dim=0)
            gt = self._load_rgb(files[-1])
            if gt.shape[-1] != self.size or gt.shape[-2] != self.size:
                gt = F.interpolate(gt.unsqueeze(0), size=(self.size, self.size), mode="bicubic", align_corners=False).squeeze(0)
            if burst.shape[-1] != self.size or burst.shape[-2] != self.size:
                burst = F.interpolate(
                    burst, size=(self.size, self.size), mode="bicubic", align_corners=False
                )
            if extra_burst.shape[-1] != self.size or extra_burst.shape[-2] != self.size:
                extra_burst = F.interpolate(
                    extra_burst, size=(self.size, self.size), mode="bicubic", align_corners=False
                )
            gt = gt * 2.0 - 1.0
            return burst, extra_burst, gt, idx

    dataset = OpticalDataset(
        dataset_root,
        split="train",
        size=image_size,
        burst_size=burst_size,
        extra_burst_size=extra_burst_size,
    )
    val_dataset = OpticalDataset(
        dataset_root,
        split="test",
        size=image_size,
        burst_size=burst_size,
        extra_burst_size=extra_burst_size,
    )
    overfit_single = False
    overfit_index = 0
    if overfit_single:
        dataset = Subset(dataset, [overfit_index])
        val_dataset = Subset(dataset, [0])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    val_index = 0
    val_burst, val_extra_burst, val_gt, _ = val_dataset[val_index]
    val_burst = val_burst.unsqueeze(0).to(device)
    val_extra_burst = val_extra_burst.unsqueeze(0).to(device)
    val_gt = val_gt.unsqueeze(0).to(device)

    # Keep logdir stable so VS Code TensorBoard can just point to it.
    log_root = Path("runs") / "vscode_tensorboard"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = log_root / f"optical_diffusion_{int(time.time())}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logdir: {log_dir}")
    print("VS Code: Command Palette -> 'TensorBoard: Start' and select this logdir.")
    # global_step can be restored when resuming

    def log_stats(tag, tensor):
        stats = (
            tensor.min().item(),
            tensor.max().item(),
            tensor.mean().item(),
            tensor.std().item(),
        )
        print(f"{tag} min={stats[0]:.4f} max={stats[1]:.4f} mean={stats[2]:.4f} std={stats[3]:.4f}")

    def conditioning_sanity_check(
        model,
        diffusion,
        burst,
        extra_features_by_t,
        device,
        epoch,
        writer=None,
        seed=1234,
        image_size=160,
    ):
        model.eval()
        with torch.no_grad():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            sample_cond = diffusion.p_sample_loop(
                model,
                (1, 3, image_size, image_size),
                model_kwargs={
                    "y": burst[:1],
                    "extra_features_by_t": extra_features_by_t[:, :1],
                },
                device=device,
            )

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            zeros_y = torch.zeros_like(burst[:1])
            zeros_extra = torch.zeros_like(extra_features_by_t[:, :1])
            sample_zero = diffusion.p_sample_loop(
                model,
                (1, 3, image_size, image_size),
                model_kwargs={
                    "y": zeros_y,
                    "extra_features_by_t": zeros_extra,
                },
                device=device,
            )

        def _stats(x):
            return (x.min().item(), x.max().item(), x.mean().item(), x.std().item())

        l2_diff = torch.norm(sample_cond - sample_zero).item()
        mad = (sample_cond - sample_zero).abs().mean().item()

        print("cond sample stats:", _stats(sample_cond))
        print("zero sample stats:", _stats(sample_zero))
        print(f"diff l2={l2_diff:.6f} mad={mad:.6f}")

        # This indicates the conditioning is likely ignored by the model.
        if l2_diff < 1e-4 and mad < 1e-4:
            print("WARNING: conditioning appears ignored (differences near zero).")

        if writer is not None:
            cond_vis = ((sample_cond + 1.0) / 2.0).clamp(0.0, 1.0)
            zero_vis = ((sample_zero + 1.0) / 2.0).clamp(0.0, 1.0)
            writer.add_image("sanity/cond", cond_vis[0], epoch)
            writer.add_image("sanity/zero", zero_vis[0], epoch)
            writer.add_scalar("sanity/l2_diff", l2_diff, epoch)
            writer.add_scalar("sanity/mad", mad, epoch)

        model.train()

    def _psnr_from_mse(mse, eps=1e-12):
        if mse <= eps:
            return 99.0
        return 10.0 * np.log10(1.0 / mse)

    def evaluate(
        model,
        diffusion,
        frame_encoder,
        implicit_aligner,
        loader,
        device,
        diffusion_steps,
        image_size,
        seed=1234,
        writer=None,
        epoch=0,
    ):
        model.eval()
        psnrs = []
        l1s = []
        sample_idx = 0
        with torch.no_grad():
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            for burst, extra_burst, gt, _sids in loader:
                burst = burst.to(device)
                extra_burst = extra_burst.to(device)
                gt = gt.to(device)
                b, n, c, h, w = extra_burst.shape
                feats = frame_encoder(extra_burst.view(b * n, c, h, w)).view(b, n, -1, h, w)
                _, fused = implicit_aligner(feats)
                extra_features = fused
                extra_features_by_t = extra_features.unsqueeze(0).expand(
                    diffusion_steps, -1, -1, -1, -1
                )
                sample = diffusion.p_sample_loop(
                    model,
                    (burst.shape[0], 3, image_size, image_size),
                    model_kwargs={
                        "y": burst,
                        "extra_features_by_t": extra_features_by_t,
                    },
                    device=device,
                )
                sample_vis = ((sample + 1.0) / 2.0).clamp(0.0, 1.0)
                gt_vis = ((gt + 1.0) / 2.0).clamp(0.0, 1.0)
                mse = F.mse_loss(sample_vis, gt_vis, reduction="mean").item()
                l1 = F.l1_loss(sample_vis, gt_vis, reduction="mean").item()
                psnrs.append(_psnr_from_mse(mse))
                l1s.append(l1)
                if writer is not None:
                    for i in range(b):
                        ref_vis = burst[i, burst.shape[1] // 2].clamp(0.0, 1.0)
                        writer.add_image(f"eval_samples/{sample_idx:04d}_ref", ref_vis, epoch)
                        writer.add_image(f"eval_samples/{sample_idx:04d}_output", sample_vis[i], epoch)
                        writer.add_image(f"eval_samples/{sample_idx:04d}_gt", gt_vis[i], epoch)
                        sample_idx += 1
        model.train()
        avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
        avg_l1 = float(np.mean(l1s)) if l1s else 0.0
        print(f"eval psnr={avg_psnr:.4f} l1={avg_l1:.6f} batches={len(psnrs)}")
        if writer is not None:
            writer.add_scalar("eval/psnr", avg_psnr, epoch)
            writer.add_scalar("eval/l1", avg_l1, epoch)
        return avg_psnr, avg_l1

    seed = 1234
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Composite loss (replaces manual charbonnier + edge)
    loss_fn = CompositeSRLoss(device=device)

    # Meta-learning settings
    use_meta_learning = True
    meta_prob = 0.5       # probability of FOMAML episode per batch
    inner_lr = 5e-5
    inner_steps = 1
    source_ratio = 0.7

    epochs = 10000
    sample_every = 50
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = 0.0
        epoch_steps = 0
        for step, (burst, extra_burst, gt, subject_ids) in enumerate(loader):
            burst = burst.to(device)
            extra_burst = extra_burst.to(device)
            gt = gt.to(device)
            subject_ids = subject_ids.to(device)

            if use_meta_learning and random.random() < meta_prob:
                # --- FOMAML meta-training episode ---
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
                # --- Standard training with composite loss ---
                b, n, c, h, w = extra_burst.shape
                feats = frame_encoder(extra_burst.view(b * n, c, h, w)).view(b, n, -1, h, w)
                _, fused = implicit_aligner(feats)
                extra_features = fused
                extra_features_by_t = extra_features.unsqueeze(0).expand(
                    diffusion_steps, -1, -1, -1, -1
                )

                t = torch.randint(0, diffusion.num_timesteps, (gt.shape[0],), device=device)
                t = t.clamp(0, diffusion.num_timesteps - 1)
                losses = diffusion.training_losses(
                    model,
                    gt,
                    t,
                    model_kwargs={
                        "y": burst,
                        "extra_features_by_t": extra_features_by_t,
                    },
                )
                diffusion_loss = losses["loss"].mean()

                noise = torch.randn_like(gt)
                x_t = diffusion.q_sample(x_start=gt, t=t, noise=noise)
                model_out = model(
                    x_t,
                    t,
                    y=burst,
                    extra_features_by_t=extra_features_by_t,
                )
                x0_pred = predict_x0(x_t, t, model_out, diffusion)

                loss, loss_dict = loss_fn(x0_pred, gt, diffusion_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.detach()

            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v, global_step)

            epoch_loss += loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val
            epoch_steps += 1
            global_step += 1

        if epoch % sample_every == 0:
            lr_vis = val_burst[0, val_burst.shape[1] // 2].clamp(0.0, 1.0)
            gt_vis = ((val_gt[0] + 1.0) / 2.0).clamp(0.0, 1.0)

            model.eval()
            with torch.no_grad():
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                vb, vn, vc, vh, vw = val_extra_burst.shape
                val_feats = frame_encoder(val_extra_burst.view(vb * vn, vc, vh, vw)).view(vb, vn, -1, vh, vw)
                _, val_fused = implicit_aligner(val_feats)
                val_extra_features = val_fused
                val_extra_features_by_t = val_extra_features.unsqueeze(0).expand(
                    diffusion_steps, -1, -1, -1, -1
                )
                sample = diffusion.p_sample_loop(
                    model,
                    (1, 3, image_size, image_size),
                    model_kwargs={
                        "y": val_burst,
                        "extra_features_by_t": val_extra_features_by_t,
                    },
                )
            model.train()
            sample_vis = ((sample + 1.0) / 2.0).clamp(0.0, 1.0)

            log_stats("lr", lr_vis)
            log_stats("gt", gt_vis)
            log_stats("sample", sample_vis[0])
            writer.add_image("image/input", lr_vis, epoch)
            writer.add_image("image/output", sample_vis[0], epoch)
            writer.add_image("image/gt", gt_vis, epoch)

            conditioning_sanity_check(
                model,
                diffusion,
                val_burst,
                val_extra_features_by_t,
                device,
                epoch,
                writer=writer,
                seed=seed,
                image_size=image_size,
            )
            evaluate(
                model,
                diffusion,
                frame_encoder,
                implicit_aligner,
                val_loader,
                device,
                diffusion_steps,
                image_size,
                seed=seed,
                writer=writer,
                epoch=epoch,
            )

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"epoch={epoch} avg_loss={avg_loss:.4f} steps={epoch_steps}")

        if epoch % 100 == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "frame_encoder": frame_encoder.state_dict(),
                "implicit_aligner": implicit_aligner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
            }
            torch.save(ckpt, "checkpoint.pt")

    writer.close()


if __name__ == "__main__":
    main()
