import torch
import torch.nn as nn
from typing import Tuple


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.reshape(1, 3) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2]  # (N, )
        z = depths.clamp(min=1e-6)
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / z[:, None] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device, dtype=means3D.dtype)
        x = cam_points[:, 0]
        y = cam_points[:, 1]
        fx = K[0, 0]
        skew = K[0, 1]
        k10 = K[1, 0]
        fy = K[1, 1]

        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 1] = skew / z
        J_proj[:, 0, 2] = -(fx * x + skew * y) / (z * z)
        J_proj[:, 1, 0] = k10 / z
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -(k10 * x + fy * y) / (z * z)
        
        # Transform covariance to camera space
        R_batch = R.unsqueeze(0).expand(N, -1, -1)
        covs_cam = torch.bmm(R_batch, torch.bmm(covs3d, R_batch.transpose(1, 2)))  # (N, 3, 3)
        
        # Project to 2D
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        eye = torch.eye(2, device=covs2D.device, dtype=covs2D.dtype).unsqueeze(0)
        covs2D = 0.5 * (covs2D + covs2D.transpose(1, 2)) + eps * eye
        
        # Compute determinant for normalization
        det = torch.linalg.det(covs2D).clamp_min(1e-8)
        inv_covs = torch.linalg.inv(covs2D)
        exponent = -0.5 * torch.einsum('nhwi,nij,nhwj->nhw', dx, inv_covs, dx)
        exponent = exponent.clamp(min=-80.0, max=20.0)
        normalizer = 1.0 / (2.0 * torch.pi * torch.sqrt(det)).view(N, 1, 1)
        gaussian = normalizer * torch.exp(exponent) ## (N, H, W)
    
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)

        valid_mask = depths > 1e-4
        means2D = means2D[valid_mask]
        covs2D = covs2D[valid_mask]
        colors = colors[valid_mask]
        opacities = opacities[valid_mask]
        depths = depths[valid_mask]

        N = means2D.shape[0]
        if N == 0:
            return torch.zeros((self.H, self.W, 3), device=means3D.device, dtype=means3D.dtype)

        indices = torch.argsort(depths, dim=0, descending=False)
        means2D = means2D[indices]
        covs2D = covs2D[indices]
        colors = colors[indices]
        opacities = opacities[indices]

        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        alphas = alphas.clamp(min=0.0, max=0.999)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones((1, self.H, self.W), device=alphas.device, dtype=alphas.dtype),
                    1.0 - alphas + 1e-10,
                ],
                dim=0,
            ),
            dim=0,
        )[:-1]
        weights = alphas * transmittance # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered.clamp(0.0, 1.0)
