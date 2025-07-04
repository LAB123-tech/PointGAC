import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from pytorch3d.transforms import euler_angles_to_matrix
from torch import nn
from torchvision import transforms


from utils.visualize import vis_points_with_label


def resample_points(points, num_points):
    if points.shape[1] > num_points:
        if num_points == 1024:
            num_samples = 1200
        elif num_points == 2048:
            num_samples = 2400
        elif num_points == 4096:
            num_samples = 4800
        elif num_points == 8192:
            num_samples = 8192
        else:
            raise NotImplementedError()
        if points.shape[1] < num_samples:
            num_samples = points.shape[1]
        _, idx = sample_farthest_points(points[:, :, :3].float(), K=num_samples, random_start_point=True)
        points = masked_gather(points, idx)
        points = points[:, torch.randperm(num_samples)[:num_points]]
        return points
    else:
        raise RuntimeError("Not enough points")


class PointCloudSubsampling(nn.Module):
    def __init__(self, num_points=1024, strategy="fps"):
        super().__init__()
        self.num_points = num_points
        self.strategy = strategy

    def forward(self, points):
        if self.strategy == "resample":
            return resample_points(points, self.num_points)
        elif self.strategy == "fps":
            _, idx = sample_farthest_points(points[:, :, :3].float(), K=self.num_points, random_start_point=True)
            data = masked_gather(points, idx)
            # point_cloud, label = data[:, :, :3], data[:, :, -1]
            # vis_points_with_label(point_cloud[0].detach().cpu().numpy(), label[0].detach().cpu().numpy(), "fps")
            return data
        elif self.strategy == "random":
            return points[:, torch.randperm(points.shape[1])[: self.num_points]]
        else:
            raise RuntimeError(f"No such subsampling strategy {self.strategy}")


class PointCloudRotation(nn.Module):
    def __init__(self, dims=None):
        super().__init__()
        # The default is \[1], meaning rotation only around the y-axis.
        if dims is None:
            dims = [1]
        self.dims = dims

    def forward(self, points):
        euler_angles = torch.zeros(3)
        for dim in self.dims:
            euler_angles[dim] = (2 * torch.pi) * torch.rand(1) - torch.pi
        R = euler_angles_to_matrix(euler_angles, "XYZ").to(points.device)
        points[:, :, :3] = points[:, :, :3] @ R.T
        # point_cloud, label = points[:, :, :3], points[:, :, -1]
        # vis_points_with_label(point_cloud[0].detach().cpu().numpy(), label[0].detach().cpu().numpy(), "rotate")
        return points


class PointCloudScale(nn.Module):
    def __init__(self, scale_low=0.9, scale_high=1.1):
        super().__init__()
        self.scale_min = scale_low
        self.scale_max = scale_high

    def forward(self, points):
        scale = (torch.rand(3, device=points.device) * (self.scale_max - self.scale_min) + self.scale_min)
        points[:, :, :3] = points[:, :, :3] * scale
        # point_cloud, label = points[:, :, :3], points[:, :, -1]
        # vis_points_with_label(point_cloud[0].detach().cpu().numpy(), label[0].detach().cpu().numpy(), "scale")
        return points


class PointCloudTranslation(nn.Module):
    def __init__(self, translate_range=0.2):
        super().__init__()
        self.translate_range = translate_range

    def forward(self, points):
        batch_size = points.size(0)
        translation = (torch.rand(batch_size, 3, device=points.device) * 2 * self.translate_range
                       - self.translate_range)
        points[:, :, :3] = points[:, :, :3] + translation.unsqueeze(1)
        # point_cloud, label = points[:, :, :3], points[:, :, -1]
        # vis_points_with_label(points[0].detach().cpu().numpy(), label[0].detach().cpu().numpy(), "translation")
        return points


class PointCloudNormalize(nn.Module):
    def __init__(self, centering: bool = True, normalize=True):
        super().__init__()
        self.centering = centering
        self.normalize = normalize

    def forward(self, points: torch.Tensor):
        if self.centering:
            points[:, :, :3] = points[:, :, :3] - torch.mean(points[:, :, :3], dim=-2, keepdim=True)
        if self.normalize:
            max_norm = torch.max(torch.norm(points[:, :, :3], dim=-1, keepdim=True),
                                 dim=-2,
                                 keepdim=True, ).values
            points[:, :, :3] = points[:, :, :3] / max_norm
        return points


def build_transformation(name):
    if name.startswith("sample"):
        return PointCloudSubsampling(num_points=int(name[-4:]))
    elif name == "rotate":
        return PointCloudRotation()
    elif name == "scale":
        return PointCloudScale()
    elif name == "translate":
        return PointCloudTranslation()
    elif name == "Normalization":
        return PointCloudNormalize()
    else:
        raise RuntimeError(f"No such transformation: {name}")


class Data_Augmentation(object):
    def __init__(self, name_list):
        self.base_transform = transforms.Compose([build_transformation(name) for name in name_list])

    def __call__(self, point_cloud_data):
        point_cloud_data = self.base_transform(point_cloud_data)
        return point_cloud_data
