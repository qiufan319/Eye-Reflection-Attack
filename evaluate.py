import argparse
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from guided_diffusion.script_util import create_model_and_diffusion_bsr


def _dynamic_create_condition(self, y, extra_features=None):
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
        self,
        channels=32,
        burst_size=5,
        window_size=7,
        fused_channels=32,
        downsample_scale=1,
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
                frame,
                size=(h // self.downsample_scale, w // self.downsample_scale),
                mode="bicubic",
                align_corners=False,
            )
            ref = F.interpolate(
                ref,
                size=(h // self.downsample_scale, w // self.downsample_scale),
                mode="bicubic",
                align_corners=False,
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
                aligned,
                size=(frame.shape[-2] * self.downsample_scale, frame.shape[-1] * self.downsample_scale),
                mode="bicubic",
                align_corners=False,
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


class OpticalDataset(Dataset):
    def __init__(self, root, split="test", size=160, burst_size=1, extra_burst_size=1):
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
            gt = F.interpolate(
                gt.unsqueeze(0),
                size=(self.size, self.size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        if burst.shape[-1] != self.size or burst.shape[-2] != self.size:
            burst = F.interpolate(
                burst, size=(self.size, self.size), mode="bicubic", align_corners=False
            )
        if extra_burst.shape[-1] != self.size or extra_burst.shape[-2] != self.size:
            extra_burst = F.interpolate(
                extra_burst, size=(self.size, self.size), mode="bicubic", align_corners=False
            )
        gt = gt * 2.0 - 1.0
        return burst, extra_burst, gt


def _psnr_from_mse(mse, eps=1e-12):
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def _to_pil(img_tensor):
    x = img_tensor.detach().cpu().clamp(0.0, 1.0)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(x)


def load_checkpoint(ckpt_path, model, frame_encoder, implicit_aligner, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "frame_encoder" in ckpt:
            frame_encoder.load_state_dict(ckpt["frame_encoder"])
        if "implicit_aligner" in ckpt:
            implicit_aligner.load_state_dict(ckpt["implicit_aligner"])
        epoch = int(ckpt.get("epoch", 0))
    else:
        model.load_state_dict(ckpt)
        epoch = 0
    return epoch


def main():
    parser = argparse.ArgumentParser(description="Evaluate burst diffusion model on test split.")
    parser.add_argument("--weights", type=str, default="checkpoint.pt")
    parser.add_argument("--dataset-root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--burst-size", type=int, default=20)
    parser.add_argument("--extra-burst-size", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, diffusion = create_model_and_diffusion_bsr(
        image_size=args.image_size,
        learn_sigma=False,
        num_channels=64,
        num_res_blocks=1,
        channel_mult="1,2,2,4",
        num_heads=2,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="40,20",
        dropout=0.0,
        diffusion_steps=args.diffusion_steps,
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
        burst_size=args.burst_size,
        num_cond_features=8,
        extra_feature_channels=32,
        extra_cond_channels=16,
    )
    model.to(device)
    model.create_condition = types.MethodType(_dynamic_create_condition, model)
    model.make_condition_feature = SimpleConditionFeature(num_features=8).to(device)

    frame_encoder = FrameEncoder(channels=32).to(device)
    implicit_aligner = ImplicitAligner(
        channels=32,
        burst_size=args.extra_burst_size,
        window_size=7,
        fused_channels=32,
        downsample_scale=1,
    ).to(device)

    ckpt_epoch = load_checkpoint(
        args.weights, model, frame_encoder, implicit_aligner, device
    )
    print(f"loaded checkpoint from {args.weights} (epoch={ckpt_epoch})")

    dataset = OpticalDataset(
        args.dataset_root,
        split=args.split,
        size=args.image_size,
        burst_size=args.burst_size,
        extra_burst_size=args.extra_burst_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = Path("eval_outputs") / f"{args.split}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving images to {out_dir}")

    model.eval()
    psnrs = []
    l1s = []
    with torch.no_grad():
        for batch_idx, (burst, extra_burst, gt) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            burst = burst.to(device)
            extra_burst = extra_burst.to(device)
            gt = gt.to(device)

            b, n, c, h, w = extra_burst.shape
            feats = frame_encoder(extra_burst.view(b * n, c, h, w)).view(b, n, -1, h, w)
            _, fused = implicit_aligner(feats)
            extra_features_by_t = fused.unsqueeze(0).expand(
                args.diffusion_steps, -1, -1, -1, -1
            )

            sample = diffusion.p_sample_loop(
                model,
                (burst.shape[0], 3, args.image_size, args.image_size),
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

            for i in range(sample_vis.shape[0]):
                global_idx = batch_idx * args.batch_size + i
                ref_vis = burst[i, burst.shape[1] // 2].clamp(0.0, 1.0)
                _to_pil(ref_vis).save(out_dir / f"{global_idx:05d}_ref.png")
                _to_pil(sample_vis[i]).save(out_dir / f"{global_idx:05d}_out.png")
                _to_pil(gt_vis[i]).save(out_dir / f"{global_idx:05d}_gt.png")

    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    avg_l1 = float(np.mean(l1s)) if l1s else 0.0
    print(f"eval psnr={avg_psnr:.4f} l1={avg_l1:.6f} batches={len(psnrs)}")


if __name__ == "__main__":
    main()
