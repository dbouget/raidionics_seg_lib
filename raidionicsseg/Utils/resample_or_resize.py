import nibabel as nib
import time
from scipy.ndimage import zoom
from nibabel.processing import resample_to_output
import numpy as np
import logging


def get_resampler(type="cpu", input_voxel_size=(1.0,1.0,1.0), target_voxel_size=(1.0,1.0,1.0), order=1):
    if type == "torch":
        try:
            import torch
        except ImportError:
            logging.warning("PyTorch not installed, cannot use TorchResampler. Defaulting to NibabelResampler!")
            return NibabelResampler(target_voxel_size=target_voxel_size, order=order)
        # return TorchResampler(input_voxel_size=input_voxel_size, target_voxel_size=target_voxel_size, order=order)
        logging.info("Running NibabelResampler even with torch acceleration")
        return NibabelResampler(target_voxel_size=target_voxel_size, order=order)
    else:
        return NibabelResampler(target_voxel_size=target_voxel_size, order=order)

class BaseResampler:
    def resample(self, volume: np.ndarray):
        raise NotImplementedError()


class NibabelResampler(BaseResampler):
    def __init__(self, input_voxel_size=(1.0, 1.0, 1.0), target_voxel_size=(1.0, 1.0, 1.0), order=1):
        self.input_voxel_size = input_voxel_size
        self.target_voxel_size = target_voxel_size
        self.interpolation_order = order

    def resample(self, volume_nib, mask_nib=None):
        resampled_volume = resample_to_output(volume_nib, self.target_voxel_size, order=self.interpolation_order)
        return resampled_volume

# class TorchResampler(BaseResampler):
#     """
#        No spline interpolation available in Torch, not possible to replicate nibabel method...
#     """
#     def __init__(self, input_voxel_size=(1.0,1.0,1.0), target_voxel_size=(1.0,1.0,1.0), order=3, device='cuda:0'):
#         from ..Utils.Torch.modules import AffineResampler3D
#         self.interpolation_order = order
#         self.target_voxel_size = target_voxel_size
#         self.model = AffineResampler3D(input_voxel_size=input_voxel_size, target_voxel_size=target_voxel_size, mode=self.interpolation_order).to(device)
#         self.device = device
#
#     def resample(self, volume_nib, mask_nib=None):
#         import torch
#         import torchio as tio
#         vol = volume_nib.get_fdata(dtype=np.float32)  # (H, W, D)
#         affine = volume_nib.affine
#         mask = None
#         if mask_nib is not None:
#             mask = mask_nib.get_fdata()[:]  # (H, W, D)
#
#         if True:
#             if self.interpolation_order == 0:
#                 input_tensor = tio.LabelMap(tensor=np.expand_dims(vol, axis=0), affine=volume_nib.affine)
#             else:
#                 input_tensor = tio.ScalarImage(tensor=np.expand_dims(vol, axis=0), affine=volume_nib.affine)
#             # img = tio.ScalarImage(tensor=np.expand_dims(vol, axis=0), affine=volume_nib.affine)
#             resampler = tio.Resample(self.target_voxel_size, image_interpolation='bspline', label_interpolation='nearest')
#             resampled = resampler(input_tensor)
#             # Second step to align to 'RAS' orientation as nibabel.resample_to_output would do.
#             # If not, there is a left/right mirroring in the axial plane, not a big deal if it happens and it would run faster
#             resampled_ras = tio.ToCanonical()(resampled)
#             data = resampled_ras.tensor.detach().cpu().numpy()[0]
#             affine = resampled_ras.affine  # (4, 4)
#
#             return data, affine
#         else:
#             # 1. Permute volume to PyTorch (D,H,W)
#             vol_torch = torch.from_numpy(vol).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,D,H,W)
#             mask_torch = None
#             if mask is not None:
#                 mask_torch = torch.from_numpy(mask).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,D,H,W)
#
#             # 2. Correctly permute affine columns to match PyTorch (D,H,W)
#             affine_permuted = affine[:, [2, 1, 0, 3]]  # D,Z->2, H,Y->1, W,X->0
#             affine_tensor = torch.from_numpy(affine_permuted).float().to(self.device)
#
#             # # 3. Send to device
#             # vol_torch = vol_torch.to(self.device)
#             # affine_tensor = affine_tensor.to(self.device)
#
#             # 4. Forward pass
#             out, affine_out, mask_out = self.model(vol_torch, affine_tensor, mask_torch)
#
#             # 5. Permute back to Nibabel axes (H,W,D)
#             out_nib = out[0, 0].permute(1, 2, 0).cpu().numpy()
#             mask_out_nib = None
#             if mask_out is not None:
#                 mask_out_nib = mask_out[0, 0].permute(1, 2, 0).cpu().numpy()
#
#             # 6. Permute affine back to Nibabel axes
#             affine_out_nib = affine_out.cpu().numpy()[:, [2, 1, 0, 3]]
#
#             return out_nib, affine_out_nib, mask_out_nib


def get_resizer(type="cpu", target_shape=(128, 128, 128), order=1):
    if type == "torch":
        try:
            import torch
        except ImportError:
            logging.warning("PyTorch not installed, cannot use TorchResampler. Defaulting to NibabelResampler!")
            return CPUResizer(target_shape=target_shape)
        return GPUResizer(target_shape=target_shape, order=order)
    else:
        return CPUResizer(target_shape=target_shape, order=order)

class BaseResizer:
    def resize(self, input, target_shape):
        raise NotImplementedError()


class CPUResizer(BaseResizer):
    def __init__(self, target_shape=(1.0,1.0,1.0), order=1):
        self.interpolation_order = order
        self.target_shape = target_shape

    def resize(self, input, target_shape):
        if len(input.shape) > len(target_shape):
            resize_ratio = tuple(np.asarray(target_shape) / np.asarray(input.shape[:-1])) + (1.,)
        else:
            resize_ratio = tuple(np.asarray(target_shape) / np.asarray(input.shape))
        data = zoom(input, list(resize_ratio), order=self.interpolation_order)
        return data


class GPUResizer(BaseResizer):
    def __init__(self, target_shape=(1.0,1.0,1.0), order=1, device='cuda:0'):
        from ..Utils.Torch.modules import TorchResizer
        self.interpolation_order = order
        self.target_shape = target_shape
        self.model = TorchResizer(order=self.interpolation_order).to(device)
        self.device = device

    def resize(self, input, target_shape):
        import torch
        if len(input.shape) > len(target_shape):
            input_t = torch.from_numpy(input).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        else:
            input_t = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).to(self.device)
        resize_t = self.model(input_t, target_shape)
        if len(input.shape) > len(target_shape):
            resize_a = resize_t[0].permute(1, 2, 3, 0).detach().cpu().numpy()
        else:
            resize_a = resize_t[0][0].detach().cpu().numpy()
        return resize_a