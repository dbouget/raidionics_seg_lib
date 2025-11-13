import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio


class AffineResampler3D(nn.Module):
    def __init__(self, input_voxel_size=(1.0,1.0,1.0), target_voxel_size=(1.0,1.0,1.0), mode=1, align_corners=True):
        super().__init__()
        self.register_buffer("input_voxel_size", torch.tensor(input_voxel_size, dtype=torch.float32))
        self.register_buffer("target_voxel_size", torch.tensor(target_voxel_size, dtype=torch.float32))
        self.mode = 'nearest' if mode == 0 else 'bilinear'
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, affine_in: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (B, C, D, H, W) tensor (PyTorch convention: Z,Y,X)
            affine_in: (4,4) tensor (nibabel affine)
        """
        B,C,D,H,W = x.shape
        device, dtype = x.device, x.dtype
        affine_in = affine_in.to(device=device, dtype=dtype)

        # --- Step 1: compute voxel sizes along tensor axes (D,H,W) ---
        voxel_sizes = torch.sqrt(torch.sum(affine_in[:3,:3]**2, dim=0))  # D,H,W

        # --- Step 2: compute rotation/orientation ---
        rotation = affine_in[:3,:3] / voxel_sizes

        # --- Step 3: compute output dimensions ---
        D_out = int(torch.ceil(D * voxel_sizes[0] / self.target_voxel_size[2]).item())
        H_out = int(torch.ceil(H * voxel_sizes[1] / self.target_voxel_size[1]).item())
        W_out = int(torch.ceil(W * voxel_sizes[2] / self.target_voxel_size[0]).item())
        print(f"✅ Output shape (PyTorch D,H,W): ({D_out}, {H_out}, {W_out})")

        # --- Step 4: build output affine ---
        affine_out = torch.eye(4, device=device, dtype=dtype)
        affine_out[:3,:3] = rotation * torch.tensor([
            self.target_voxel_size[2],  # D/Z
            self.target_voxel_size[1],  # H/Y
            self.target_voxel_size[0]   # W/X
        ], device=device, dtype=dtype)
        affine_out[:3,3] = affine_in[:3,3]

        # --- Step 5: build output grid ---
        dz = torch.linspace(0, D_out-1, D_out, device=device, dtype=dtype)
        dy = torch.linspace(0, H_out-1, H_out, device=device, dtype=dtype)
        dx = torch.linspace(0, W_out-1, W_out, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing='ij')
        ones = torch.ones_like(xx)
        grid_out = torch.stack([zz, yy, xx, ones], dim=-1)

        # --- Step 6: map output grid → input coordinates ---
        T = torch.linalg.inv(affine_in) @ affine_out
        grid_in = torch.einsum('ij,dhwi->dhwj', T.T, grid_out)[..., :3]

        # --- Step 7: normalize grid for grid_sample ---
        grid_norm = torch.empty_like(grid_in)
        grid_norm[...,0] = 2.0 * grid_in[...,0] / max(W-1,1) - 1.0
        grid_norm[...,1] = 2.0 * grid_in[...,1] / max(H-1,1) - 1.0
        grid_norm[...,2] = 2.0 * grid_in[...,2] / max(D-1,1) - 1.0
        grid_norm = grid_norm.unsqueeze(0)  # batch dim

        # --- Step 8: resample ---
        x_resampled = F.grid_sample(
            x, grid_norm, mode=self.mode,
            padding_mode='border', align_corners=self.align_corners
        )
        mask_resampled = None
        if mask is not None:
            mask_resampled = F.grid_sample(
                mask, grid_norm, mode='nearest',
                padding_mode='border', align_corners=self.align_corners
            )
        return x_resampled, affine_out, mask_resampled


class TorchResizer(nn.Module):
    """
    """

    def __init__(self, order=1, align_corners=False):
        """
        Args:
            mode (str): Interpolation mode: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
            align_corners (bool): Align corners flag for linear modes.
        """
        super().__init__()
        self.mode = 'nearest' if order == 0 else 'bilinear'
        self.align_corners = align_corners

    def forward(self, volume: torch.Tensor, target_shape) -> torch.Tensor:
        """
        Resample a tensor to a target shape.

        Args:
            volume (torch.Tensor): Input tensor (N, C, D, H, W) or (N, C, H, W).
                                   Must be on CUDA for GPU use.
            target_shape (tuple/list): Desired spatial shape (D, H, W) or (H, W).

        Returns:
            torch.Tensor: Resampled tensor on the same device.
        """
        if not torch.is_tensor(volume):
            raise TypeError("volume must be a torch.Tensor")
        if volume.ndim not in (4, 5):
            raise ValueError("Expected tensor of shape (N,C,H,W) or (N,C,D,H,W)")
        if not volume.is_cuda:
            raise ValueError("Tensor must be on GPU for GPU resampling")

        if volume.ndim == 5:
            self.mode = 'trilinear'

        return F.interpolate(
            volume,
            size=target_shape,
            mode=self.mode,
            align_corners=self.align_corners if 'linear' in self.mode else None
        )