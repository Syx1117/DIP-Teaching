"""
Task 1: Bundle Adjustment from scratch with PyTorch.

Expected data layout:
    data/
    |-- points2d.npz         # keys view_000 ... view_049, each (N, 3): x, y, visibility
    |-- points3d_colors.npy  # (N, 3), RGB colors

Example:
    python outputs/ba_task1_pytorch.py ^
        --data_dir path/to/Assignments/03_BundleAdjustment/data ^
        --out_dir outputs/ba_task1 ^
        --steps 5000 ^
        --batch_points 4096

Outputs:
    reconstructed_points.obj
    reconstructed_points.ply
    ba_result.npz
    loss_curve.png, loss_log.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BAConfig:
    image_width: int = 1024
    image_height: int = 1024
    init_fov_deg: float = 60.0
    init_camera_distance: float = 2.5
    init_yaw_range_deg: float = 70.0
    steps: int = 5000
    batch_points: int = 4096
    lr_points: float = 1e-2
    lr_pose: float = 3e-3
    lr_focal: float = 1e-3
    pose_prior_weight: float = 1e-4
    point_center_weight: float = 1e-4
    depth_weight: float = 1e-2
    min_negative_depth: float = 0.05
    log_every: int = 50
    eval_every: int = 250
    freeze_first_camera: bool = False
    seed: int = 0


def sort_view_keys(keys: list[str]) -> list[str]:
    def key_fn(key: str) -> tuple[int, str]:
        match = re.search(r"(\d+)", key)
        return (int(match.group(1)) if match else 10**9, key)

    return sorted(keys, key=key_fn)


def load_observations(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points2d_path = data_dir / "points2d.npz"
    colors_path = data_dir / "points3d_colors.npy"
    if not points2d_path.exists():
        raise FileNotFoundError(f"Cannot find {points2d_path}")
    if not colors_path.exists():
        raise FileNotFoundError(f"Cannot find {colors_path}")

    packed = np.load(points2d_path)
    view_keys = sort_view_keys(list(packed.keys()))
    points2d = np.stack([packed[key].astype(np.float32) for key in view_keys], axis=0)
    xy = points2d[..., :2]
    visibility = points2d[..., 2].astype(np.float32)
    colors = np.load(colors_path).astype(np.float32)
    return xy, visibility, colors


def focal_from_fov(image_height: int, fov_deg: float) -> float:
    fov_rad = math.radians(fov_deg)
    return image_height / (2.0 * math.tan(fov_rad / 2.0))


def inverse_softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Unknown axis {axis}")

    return torch.stack(flat, dim=-1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix_xyz(euler_angles: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to rotation matrices for column vectors."""
    rx = axis_angle_rotation("X", euler_angles[..., 0])
    ry = axis_angle_rotation("Y", euler_angles[..., 1])
    rz = axis_angle_rotation("Z", euler_angles[..., 2])
    return rx @ ry @ rz


def initialize_points_from_observations(
    observations_xy: np.ndarray,
    visibility: np.ndarray,
    focal: float,
    camera_distance: float,
    image_width: int,
    image_height: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cx = image_width * 0.5
    cy = image_height * 0.5
    vis_sum = visibility.sum(axis=0)
    safe_vis_sum = np.maximum(vis_sum, 1.0)

    mean_u = (observations_xy[..., 0] * visibility).sum(axis=0) / safe_vis_sum
    mean_v = (observations_xy[..., 1] * visibility).sum(axis=0) / safe_vis_sum
    x = (mean_u - cx) * camera_distance / focal
    y = (cy - mean_v) * camera_distance / focal
    z = rng.normal(loc=0.0, scale=0.03, size=x.shape)

    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    invisible = vis_sum <= 0
    if invisible.any():
        points[invisible] = rng.normal(loc=0.0, scale=0.05, size=(int(invisible.sum()), 3))
    return points


def initialize_camera_poses(num_views: int, cfg: BAConfig) -> tuple[np.ndarray, np.ndarray]:
    euler = np.zeros((num_views, 3), dtype=np.float32)
    if cfg.init_yaw_range_deg > 0 and num_views > 1:
        yaw = np.linspace(
            -math.radians(cfg.init_yaw_range_deg),
            math.radians(cfg.init_yaw_range_deg),
            num_views,
            dtype=np.float32,
        )
        euler[:, 1] = yaw

    translations = np.zeros((num_views, 3), dtype=np.float32)
    translations[:, 2] = -float(cfg.init_camera_distance)
    return euler, translations


class BundleAdjustmentModel(nn.Module):
    def __init__(
        self,
        init_points3d: np.ndarray,
        init_euler: np.ndarray,
        init_translations: np.ndarray,
        init_focal: float,
        image_width: int,
        image_height: int,
    ) -> None:
        super().__init__()
        self.points3d = nn.Parameter(torch.from_numpy(init_points3d).float())
        self.euler_angles = nn.Parameter(torch.from_numpy(init_euler).float())
        self.translations = nn.Parameter(torch.from_numpy(init_translations).float())
        self.raw_focal = nn.Parameter(torch.tensor(inverse_softplus(init_focal), dtype=torch.float32))

        self.register_buffer("init_euler", torch.from_numpy(init_euler).float())
        self.register_buffer("init_translations", torch.from_numpy(init_translations).float())
        self.register_buffer("init_focal", torch.tensor(float(init_focal), dtype=torch.float32))

        self.cx = image_width * 0.5
        self.cy = image_height * 0.5

    @property
    def focal(self) -> torch.Tensor:
        return F.softplus(self.raw_focal) + 1e-6

    def forward(self, point_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = self.points3d[point_ids]
        rotation = euler_angles_to_matrix_xyz(self.euler_angles)
        camera_points = torch.einsum("vij,nj->vni", rotation, points)
        camera_points = camera_points + self.translations[:, None, :]

        x = camera_points[..., 0]
        y = camera_points[..., 1]
        z = camera_points[..., 2]
        z_safe = torch.where(z > -1e-4, torch.full_like(z, -1e-4), z)

        u = -self.focal * x / z_safe + self.cx
        v = self.focal * y / z_safe + self.cy
        projected = torch.stack([u, v], dim=-1)
        return projected, z

    def pose_prior_loss(self) -> torch.Tensor:
        pose_loss = (self.euler_angles - self.init_euler).pow(2).mean()
        trans_loss = (self.translations - self.init_translations).pow(2).mean()
        focal_loss = torch.log(self.focal / self.init_focal).pow(2)
        return pose_loss + trans_loss + focal_loss

    def keep_gauge_fixed(self) -> None:
        with torch.no_grad():
            self.euler_angles[0].copy_(self.init_euler[0])
            self.translations[0].copy_(self.init_translations[0])


def reprojection_loss(
    projected_xy: torch.Tensor,
    observed_xy: torch.Tensor,
    visibility: torch.Tensor,
    image_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    visible = visibility > 0.5
    if not visible.any():
        zero = projected_xy.sum() * 0.0
        return zero, zero.detach()

    pixel_error = projected_xy - observed_xy
    sq_error = pixel_error.pow(2).sum(dim=-1)
    mean_sq_error = sq_error[visible].mean()
    normalized_loss = mean_sq_error / (image_size * image_size)
    rmse_pixels = torch.sqrt(mean_sq_error.detach())
    return normalized_loss, rmse_pixels


def full_rmse(
    model: BundleAdjustmentModel,
    observations_xy: torch.Tensor,
    visibility: torch.Tensor,
    image_size: float,
    chunk_points: int = 4096,
) -> float:
    num_points = observations_xy.shape[1]
    sq_error_sum = 0.0
    visible_count = 0.0
    with torch.no_grad():
        for start in range(0, num_points, chunk_points):
            end = min(start + chunk_points, num_points)
            ids = torch.arange(start, end, device=observations_xy.device)
            pred, _ = model(ids)
            pixel_error = pred - observations_xy[:, start:end]
            sq_error = pixel_error.pow(2).sum(dim=-1)
            visible = visibility[:, start:end] > 0.5
            sq_error_sum += float(sq_error[visible].sum().cpu())
            visible_count += float(visible.sum().cpu())
    if visible_count == 0:
        return float("nan")
    return math.sqrt(sq_error_sum / visible_count)


def save_obj(path: Path, points3d: np.ndarray, colors: np.ndarray) -> None:
    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)

    with path.open("w", encoding="utf-8") as file:
        for point, color in zip(points3d, colors):
            file.write(
                "v "
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
            )


def save_ply(path: Path, points3d: np.ndarray, colors: np.ndarray) -> None:
    colors = colors.astype(np.float32)
    if colors.max() <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points3d)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points3d, colors):
            file.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def save_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plotting loss curve because matplotlib is unavailable: {exc}")
        return

    steps = [row["step"] for row in history]
    train_rmse = [row["train_rmse_px"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(steps, train_rmse, label="train mini-batch RMSE")
    eval_steps = [row["step"] for row in history if not math.isnan(row["full_rmse_px"])]
    eval_rmse = [row["full_rmse_px"] for row in history if not math.isnan(row["full_rmse_px"])]
    if eval_steps:
        plt.plot(eval_steps, eval_rmse, "o-", label="full RMSE")
    plt.xlabel("Optimization step")
    plt.ylabel("Reprojection RMSE (pixels)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_loss_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_ba(
    observations_xy_np: np.ndarray,
    visibility_np: np.ndarray,
    colors_np: np.ndarray,
    cfg: BAConfig,
    out_dir: Path,
    device: torch.device,
) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_views, num_points, _ = observations_xy_np.shape
    init_focal = focal_from_fov(cfg.image_height, cfg.init_fov_deg)
    init_points = initialize_points_from_observations(
        observations_xy_np,
        visibility_np,
        init_focal,
        cfg.init_camera_distance,
        cfg.image_width,
        cfg.image_height,
        cfg.seed,
    )
    init_euler, init_translations = initialize_camera_poses(num_views, cfg)

    model = BundleAdjustmentModel(
        init_points,
        init_euler,
        init_translations,
        init_focal,
        cfg.image_width,
        cfg.image_height,
    ).to(device)

    observations_xy = torch.from_numpy(observations_xy_np).float().to(device)
    visibility = torch.from_numpy(visibility_np).float().to(device)
    image_size = float(max(cfg.image_width, cfg.image_height))

    optimizer = torch.optim.Adam(
        [
            {"params": [model.points3d], "lr": cfg.lr_points},
            {"params": [model.euler_angles, model.translations], "lr": cfg.lr_pose},
            {"params": [model.raw_focal], "lr": cfg.lr_focal},
        ]
    )

    history: list[dict[str, float]] = []
    all_point_ids = torch.arange(num_points, device=device)
    print(f"Views: {num_views}, points: {num_points}, visible observations: {int(visibility.sum().item())}")
    print(f"Initial focal: {init_focal:.3f}, device: {device}")

    for step in range(cfg.steps + 1):
        if cfg.batch_points >= num_points:
            point_ids = all_point_ids
        else:
            point_ids = torch.randperm(num_points, device=device)[: cfg.batch_points]

        projected, depth = model(point_ids)
        observed_batch = observations_xy[:, point_ids]
        visibility_batch = visibility[:, point_ids]
        loss, train_rmse = reprojection_loss(projected, observed_batch, visibility_batch, image_size)

        if cfg.pose_prior_weight > 0:
            loss = loss + cfg.pose_prior_weight * model.pose_prior_loss()
        if cfg.point_center_weight > 0:
            loss = loss + cfg.point_center_weight * model.points3d.mean(dim=0).pow(2).sum()
        if cfg.depth_weight > 0:
            depth_penalty = F.relu(depth + cfg.min_negative_depth).pow(2).mean()
            loss = loss + cfg.depth_weight * depth_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        if cfg.freeze_first_camera:
            model.keep_gauge_fixed()

        should_eval = step % cfg.eval_every == 0 or step == cfg.steps
        full_eval = (
            full_rmse(model, observations_xy, visibility, image_size, cfg.batch_points)
            if should_eval
            else float("nan")
        )

        if step % cfg.log_every == 0 or step == cfg.steps:
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "train_rmse_px": float(train_rmse.cpu()),
                "full_rmse_px": float(full_eval),
                "focal": float(model.focal.detach().cpu()),
            }
            history.append(row)
            eval_text = f", full_rmse={full_eval:.3f}px" if not math.isnan(full_eval) else ""
            print(
                f"[{step:05d}/{cfg.steps}] "
                f"loss={row['loss']:.6f}, batch_rmse={row['train_rmse_px']:.3f}px"
                f"{eval_text}, f={row['focal']:.3f}"
            )

    points_out = model.points3d.detach().cpu().numpy()
    euler_out = model.euler_angles.detach().cpu().numpy()
    trans_out = model.translations.detach().cpu().numpy()
    focal_out = float(model.focal.detach().cpu())

    save_obj(out_dir / "reconstructed_points.obj", points_out, colors_np)
    save_ply(out_dir / "reconstructed_points.ply", points_out, colors_np)
    np.savez(
        out_dir / "ba_result.npz",
        points3d=points_out,
        euler_angles=euler_out,
        translations=trans_out,
        focal=focal_out,
    )
    write_loss_csv(out_dir / "loss_log.csv", history)
    save_loss_curve(out_dir / "loss_curve.png", history)
    print(f"Saved results to {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Bundle Adjustment for Assignment 3 Task 1")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ba_task1"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image_width", type=int, default=1024)
    parser.add_argument("--image_height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_points", type=int, default=4096)
    parser.add_argument("--init_fov_deg", type=float, default=60.0)
    parser.add_argument("--init_camera_distance", type=float, default=2.5)
    parser.add_argument("--init_yaw_range_deg", type=float, default=70.0)
    parser.add_argument("--lr_points", type=float, default=1e-2)
    parser.add_argument("--lr_pose", type=float, default=3e-3)
    parser.add_argument("--lr_focal", type=float, default=1e-3)
    parser.add_argument("--pose_prior_weight", type=float, default=1e-4)
    parser.add_argument("--point_center_weight", type=float, default=1e-4)
    parser.add_argument("--depth_weight", type=float, default=1e-2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--freeze_first_camera", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg = BAConfig(
        image_width=args.image_width,
        image_height=args.image_height,
        init_fov_deg=args.init_fov_deg,
        init_camera_distance=args.init_camera_distance,
        init_yaw_range_deg=args.init_yaw_range_deg,
        steps=args.steps,
        batch_points=args.batch_points,
        lr_points=args.lr_points,
        lr_pose=args.lr_pose,
        lr_focal=args.lr_focal,
        pose_prior_weight=args.pose_prior_weight,
        point_center_weight=args.point_center_weight,
        depth_weight=args.depth_weight,
        log_every=args.log_every,
        eval_every=args.eval_every,
        freeze_first_camera=args.freeze_first_camera,
        seed=args.seed,
    )

    observations_xy, visibility, colors = load_observations(args.data_dir)
    train_ba(observations_xy, visibility, colors, cfg, args.out_dir, device)


if __name__ == "__main__":
    main()
