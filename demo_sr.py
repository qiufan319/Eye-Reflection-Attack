"""
Quick smoke-test for the 4x SR diffusion pipeline.
HR=64, LR=16, synthetic data — no real dataset needed.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.script_util import create_model_and_diffusion_bsr


def _dynamic_create_condition(self, y, extra_features=None):
    y = self.make_condition_feature(y)
    if extra_features is not None:
        if self.extra_feature_projector is None:
            raise ValueError(
                "extra_features provided but extra_feature_channels/extra_cond_channels are not set"
            )
        if extra_features.shape[-2:] != y.shape[-2:]:
            extra_features = F.interpolate(extra_features, size=y.shape[-2:], mode="bicubic")
        extra_features = self.extra_feature_projector(extra_features)
        y = torch.cat([y, extra_features], dim=1)
    elif self.extra_cond_channels > 0:
        zeros = torch.zeros(
            y.shape[0], self.extra_cond_channels, y.shape[2], y.shape[3],
            device=y.device, dtype=y.dtype,
        )
        y = torch.cat([y, zeros], dim=1)

    y_lr_size = y.shape[-1]
    if hasattr(self, "cond_sr_upsampler") and y_lr_size < self.image_size:
        y_hr = self.cond_sr_upsampler(y)
    else:
        y_hr = F.interpolate(y, size=(self.image_size, self.image_size), mode="bicubic")

    sizes = [self.image_size // (2**i) for i in range(len(self.channel_mult))]
    sizes = sorted({int(s) for s in sizes if s > 0})
    y_dic = {}
    for s in sizes:
        if s == y_lr_size:
            y_dic[str(s)] = y
        elif s < y_lr_size:
            y_dic[str(s)] = F.interpolate(y, size=(s, s), mode="bicubic")
        elif s == y_hr.shape[-1]:
            y_dic[str(s)] = y_hr
        else:
            y_dic[str(s)] = F.interpolate(y_hr, size=(s, s), mode="bicubic")
    return y_dic


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Small demo params: HR=64, LR=16
    hr_size    = 64
    burst_size = 4
    extra_burst_size = 4
    batch_size = 2
    diffusion_steps = 20
    num_cond_features = 8
    extra_cond_channels = 16

    print(f"HR={hr_size}, LR={hr_size//4}, burst={burst_size}")

    model, diffusion = create_model_and_diffusion_bsr(
        image_size=hr_size,
        learn_sigma=False,
        num_channels=32,
        num_res_blocks=1,
        channel_mult="1,2,4",
        num_heads=2,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        diffusion_steps=diffusion_steps,
        noise_schedule="sigmoid",
        timestep_respacing="ddim10",
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
        num_cond_features=num_cond_features,
        extra_feature_channels=32,
        extra_cond_channels=extra_cond_channels,
    )
    model.to(device)
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

    model.make_condition_feature = SimpleConditionFeature(num_features=num_cond_features).to(device)

    class ConditionSRUpsampler(nn.Module):
        """4x pixel-shuffle upsampler for LR condition features (two 2x stages)."""
        def __init__(self, in_channels):
            super().__init__()
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(),
            )
        def forward(self, x):
            return self.stage2(self.stage1(x))

    cond_channels = burst_size * num_cond_features + extra_cond_channels  # 4*8+16=48
    model.cond_sr_upsampler = ConditionSRUpsampler(in_channels=cond_channels).to(device)
    print(f"cond_sr_upsampler in_channels={cond_channels}")

    class FrameEncoder(nn.Module):
        def __init__(self, channels=32):
            super().__init__()
            self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.act = nn.SiLU()
        def forward(self, x):
            return self.act(self.conv2(self.act(self.conv1(x))))

    class ImplicitAligner(nn.Module):
        def __init__(self, channels=32, burst_size=4, window_size=7, fused_channels=32):
            super().__init__()
            self.window_size = window_size
            self.padding = window_size // 2
            self.fuse = nn.Conv2d(burst_size * channels, fused_channels, kernel_size=1)
        def _align_one(self, frame, ref):
            b, c, h, w = frame.shape
            patches = F.unfold(frame, kernel_size=self.window_size, padding=self.padding)
            patches = patches.view(b, c, self.window_size * self.window_size, h * w)
            ref_center = ref.view(b, c, h * w).unsqueeze(2)
            attn = F.softmax((patches * ref_center).sum(dim=1), dim=1)
            aligned = (patches * attn.unsqueeze(1)).sum(dim=2).view(b, c, h, w)
            return aligned
        def forward(self, feats):
            b, n, c, h, w = feats.shape
            ref = feats[:, n // 2]
            aligned = torch.stack([self._align_one(feats[:, i], ref) for i in range(n)], dim=1)
            fused = self.fuse(aligned.view(b, n * c, h, w))
            return aligned, fused

    frame_encoder = FrameEncoder(channels=32).to(device)
    implicit_aligner = ImplicitAligner(channels=32, burst_size=extra_burst_size, fused_channels=32).to(device)

    lr_size = hr_size // 4  # 16

    # ---- synthetic batch ----
    burst       = torch.rand(batch_size, burst_size, 3, lr_size, lr_size, device=device)
    extra_burst = torch.rand(batch_size, extra_burst_size, 3, lr_size, lr_size, device=device)
    gt          = torch.rand(batch_size, 3, hr_size, hr_size, device=device) * 2 - 1  # [-1,1]

    print(f"burst={tuple(burst.shape)}  extra_burst={tuple(extra_burst.shape)}  gt={tuple(gt.shape)}")

    # ---- encode extra features ----
    b, n, c, h, w = extra_burst.shape
    feats = frame_encoder(extra_burst.view(b * n, c, h, w)).view(b, n, -1, h, w)
    _, fused = implicit_aligner(feats)
    extra_features_by_t = fused.unsqueeze(0).expand(diffusion_steps, -1, -1, -1, -1)

    print(f"extra_features_by_t={tuple(extra_features_by_t.shape)}")

    # ---- training_losses forward pass ----
    t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    losses = diffusion.training_losses(
        model, gt, t,
        model_kwargs={"y": burst, "extra_features_by_t": extra_features_by_t},
    )
    loss = losses["loss"].mean()
    print(f"training_losses OK  loss={loss.item():.4f}")

    # ---- manual model forward to check output shape ----
    noise = torch.randn_like(gt)
    x_t = diffusion.q_sample(x_start=gt, t=t, noise=noise)
    model_out = model(x_t, t, y=burst, extra_features_by_t=extra_features_by_t)
    print(f"model input x_t={tuple(x_t.shape)}  output={tuple(model_out.shape)}")
    assert model_out.shape == gt.shape, f"output shape mismatch: {model_out.shape} != {gt.shape}"

    # ---- sampling (ddim10, 1 sample to keep it fast) ----
    print("running p_sample_loop (ddim10, 1 image)...")
    model.eval()
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (1, 3, hr_size, hr_size),
            model_kwargs={
                "y": burst[:1],
                "extra_features_by_t": extra_features_by_t[:, :1],
            },
            device=device,
        )
    print(f"sample={tuple(sample.shape)}  min={sample.min():.3f}  max={sample.max():.3f}")
    assert sample.shape == (1, 3, hr_size, hr_size)

    print("\nAll checks passed. SR pipeline (LR=16->HR=64) is working.")


if __name__ == "__main__":
    main()
