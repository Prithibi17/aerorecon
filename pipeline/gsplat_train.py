"""Small, deterministic gsplat trainer for an exported AeroRecon camera set."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from plyfile import PlyData, PlyElement


def logit(value):
    value = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(value / (1 - value))


def ssim(a, b, mask):
    # Differentiable local SSIM; mask removes sky from the final average.
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mu_x, mu_y = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    vx = F.avg_pool2d(x * x, 7, 1, 3) - mu_x.square()
    vy = F.avg_pool2d(y * y, 7, 1, 3) - mu_y.square()
    vxy = F.avg_pool2d(x * y, 7, 1, 3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + .01 ** 2) * (2 * vxy + .03 ** 2) /
             ((mu_x.square() + mu_y.square() + .01 ** 2) * (vx + vy + .03 ** 2))).mean(1)[0]
    return (score * mask).sum() / mask.sum().clamp_min(1)


def write_gaussian_ply(path, means, colors, scales, opacities, quats):
    n = len(means)
    fields = [(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz",
              "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3")]
    data = np.empty(n, dtype=fields)
    for i, name in enumerate(("x", "y", "z")): data[name] = means[:, i]
    for name in ("nx", "ny", "nz"): data[name] = 0
    sh0 = (colors - .5) / .28209479177387814
    for i in range(3): data[f"f_dc_{i}"] = sh0[:, i]
    data["opacity"] = logit(opacities)
    for i in range(3): data[f"scale_{i}"] = np.log(np.maximum(scales[:, i], 1e-8))
    for i in range(4): data[f"rot_{i}"] = quats[:, i]
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def load_view(root, camera, device):
    rgb = torch.from_numpy(np.asarray(Image.open(root / "images" / camera["name"]).convert("RGB"), dtype=np.float32).copy() / 255).to(device)
    mask_name = Path(camera["name"]).with_suffix(".jpg.png").name
    mask_path = root / "masks" / mask_name
    mask = torch.from_numpy(np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32).copy() / 255).to(device)
    return rgb, (mask > .5).float()


def render(params, camera, device):
    colors, alphas, _ = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), params["log_scales"].exp(),
        params["opacity_logits"].sigmoid().squeeze(-1), params["color_logits"].sigmoid(),
        torch.tensor(camera["world_to_camera"], dtype=torch.float32, device=device)[None],
        torch.tensor(camera["K"], dtype=torch.float32, device=device)[None],
        camera["width"], camera["height"], packed=True, rasterize_mode="antialiased")
    alpha = alphas[0, ..., 0]
    background = torch.tensor([.55, .72, .82], device=device)
    return colors[0] + (1 - alpha[..., None]) * background, alpha


def evaluate(root, cameras, params, device, render_dir):
    scores = []
    render_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for index, camera in enumerate(cameras):
            target, mask = load_view(root, camera, device)
            predicted, _ = render(params, camera, device)
            mse = (((predicted - target).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1)).item()
            scores.append(-10 * math.log10(max(mse, 1e-10)))
            if index < 6:
                pred = (predicted.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                truth = (target.cpu().numpy() * 255).astype(np.uint8)
                panel = Image.new("RGB", (camera["width"] * 2, camera["height"] + 28), "#102128")
                panel.paste(Image.fromarray(truth), (0, 28)); panel.paste(Image.fromarray(pred), (camera["width"], 28))
                draw = ImageDraw.Draw(panel); draw.text((8, 7), f"VIDEO FRAME (left)     GAUSSIAN RENDER (right)     PSNR {scores[-1]:.2f} dB", fill="white")
                panel.save(render_dir / f"holdout_{index:02d}.jpg", quality=92)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()
    root, out = Path(args.dataset), Path(args.out)
    device = torch.device("cuda")
    if not torch.cuda.is_available(): raise RuntimeError("A CUDA GPU is required for gsplat training")
    torch.manual_seed(42); np.random.seed(42)
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    cameras = dataset["cameras"]
    train = [c for i, c in enumerate(cameras) if i % 8]
    holdout = [c for i, c in enumerate(cameras) if not i % 8]
    initial = np.load(root / "initial_gaussians.npz")
    means = initial["means"].astype(np.float32); colors = initial["colors"].astype(np.float32)
    extent = np.quantile(np.linalg.norm(means - np.median(means, axis=0), axis=1), .9)
    scale = max(float(extent / math.sqrt(len(means))), 1e-4)
    parameters = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(means).to(device)),
        "quats": torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]], device=device).repeat(len(means), 1)),
        "log_scales": torch.nn.Parameter(torch.full((len(means), 3), math.log(scale), device=device)),
        "opacity_logits": torch.nn.Parameter(torch.full((len(means), 1), float(logit(.72)), device=device)),
        "color_logits": torch.nn.Parameter(torch.from_numpy(logit(np.clip(colors, .02, .98))).to(device)),
    })
    optimizer = torch.optim.Adam([
        {"params": [parameters["means"]], "lr": 5e-4}, {"params": [parameters["quats"]], "lr": 2e-4},
        {"params": [parameters["log_scales"]], "lr": 1e-3}, {"params": [parameters["opacity_logits"]], "lr": 5e-3},
        {"params": [parameters["color_logits"]], "lr": 8e-3}], eps=1e-15)
    anchor = parameters['means'].detach().clone()
    losses = []
    for step in range(args.steps):
        camera = train[step % len(train)]
        target, mask = load_view(root, camera, device)
        prediction, alpha = render(parameters, camera, device)
        l1 = ((prediction - target).abs().mean(-1) * mask).sum() / mask.sum().clamp_min(1)
        structural = 1 - ssim(prediction, target, mask)
        uncovered = ((1 - alpha) * mask).sum() / mask.sum().clamp_min(1)
        scale_reg = parameters["log_scales"].exp().mean() / max(extent, 1e-5)
        sky = 1 - mask
        sky_opacity = (alpha * sky).sum() / sky.sum().clamp_min(1)
        displacement = (parameters['means'] - anchor).square().mean() / max(scale * scale, 1e-8)
        loss = .78 * l1 + .2 * structural + .02 * uncovered + .25 * sky_opacity + .002 * displacement + 1e-4 * scale_reg
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            parameters['log_scales'].clamp_(math.log(scale * .1), math.log(scale * 3))
        losses.append(float(loss.detach()))
        if step % 25 == 0 or step + 1 == args.steps:
            print(f"step {step + 1}/{args.steps} loss={losses[-1]:.5f} gaussians={len(means)}", flush=True)
        if step and step % args.eval_every == 0:
            score = evaluate(root, holdout[:2], parameters, device, out / "validation_preview")
            print(f"validation step={step} psnr={score:.2f}dB", flush=True)

    psnr = evaluate(root, holdout, parameters, device, out / "validation_renders")
    values = {key: value.detach().cpu().numpy() for key, value in parameters.items()}
    final_colors = 1 / (1 + np.exp(-values["color_logits"]))
    final_scales = np.exp(values["log_scales"])
    final_opacity = 1 / (1 + np.exp(-values["opacity_logits"][:, 0]))
    final_quats = values["quats"] / np.linalg.norm(values["quats"], axis=1, keepdims=True).clip(1e-8)
    write_gaussian_ply(out / "gaussians.ply", values["means"], final_colors, final_scales, final_opacity, final_quats)
    torch.save({key: value.detach().cpu() for key, value in parameters.items()}, out / "gaussians.pt")
    keep = np.argsort(final_opacity)[-min(100000, len(means)):]
    viewer = {"positions": values["means"][keep].astype(float).ravel().tolist(),
              "colors": np.clip(final_colors[keep] * 255, 0, 255).astype(np.uint8).ravel().tolist(),
              "scales": final_scales[keep].max(1).astype(float).tolist(),
              "opacities": final_opacity[keep].astype(float).tolist()}
    (out / "gaussians_viewer.json").write_text(json.dumps(viewer, separators=(",", ":")), encoding="utf-8")
    cameras_out = [{"name": c["name"], "centre": c["centre"], "world_to_camera": c["world_to_camera"]} for c in cameras]
    cloud = {**viewer, "cameras": cameras_out, "total_points": len(means), "displayed_points": len(keep),
             "units": "arbitrary", "geometry_provenance": "gsplat/PyTorch optimized appearance",
             "surface_available": True, "gaussian_splats": True, "ground_aligned": dataset.get('ground_aligned', dataset.get('coordinate_system')=='ground-aligned-relative')}
    (out / "viewer.json").write_text(json.dumps(cloud, separators=(",", ":")), encoding="utf-8")
    metrics = {"training_steps": args.steps, "gaussians": len(means), "viewer_gaussians": len(keep),
               "training_views": len(train), "holdout_views": len(holdout), "holdout_psnr_db": psnr,
               "final_loss": float(np.mean(losses[-50:])), "pytorch_version": torch.__version__,
               "gpu": torch.cuda.get_device_name(0), "gsplat_backend": "CUDA"}
    (out / "gsplat_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
