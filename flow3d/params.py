import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow3d.transforms import cont_6d_to_rmat


class GaussianParams(nn.Module):
    def __init__(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        motion_coefs: torch.Tensor | None = None,
        scene_center: torch.Tensor | None = None,
        scene_scale: torch.Tensor | float = 1.0,
    ):
        super().__init__()
        if not check_gaussian_sizes(
            means, quats, scales, colors, opacities, motion_coefs
        ):
            import ipdb

            ipdb.set_trace()
        params_dict = {
            "means": nn.Parameter(means),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(scales),
            "colors": nn.Parameter(colors),
            "opacities": nn.Parameter(opacities),
        }
        if motion_coefs is not None:
            params_dict["motion_coefs"] = nn.Parameter(motion_coefs)
        self.params = nn.ParameterDict(params_dict)
        self.quat_activation = lambda x: F.normalize(x, dim=-1, p=2)
        self.color_activation = torch.sigmoid
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)

        if scene_center is None:
            scene_center = torch.zeros(3, device=means.device)
        # self.scene_center = scene_center
        # self.scene_scale = scene_scale
        self.register_buffer("scene_center", scene_center)
        self.register_buffer("scene_scale", torch.as_tensor(scene_scale))

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        req_keys = ["means", "quats", "scales", "colors", "opacities"]
        assert all(f"{prefix}{k}" in state_dict for k in req_keys)
        args = {
            "motion_coefs": None,
            "scene_center": torch.zeros(3),
            "scene_scale": torch.tensor(1.0),
        }
        for k in req_keys + list(args.keys()):
            if f"{prefix}{k}" in state_dict:
                args[k] = state_dict[f"{prefix}{k}"]
        return GaussianParams(**args)

    @property
    def num_gaussians(self) -> int:
        return self.params["means"].shape[0]

    def get_colors(self) -> torch.Tensor:
        return self.color_activation(self.params["colors"])

    def get_scales(self) -> torch.Tensor:
        return self.scale_activation(self.params["scales"])

    def get_opacities(self) -> torch.Tensor:
        return self.opacity_activation(self.params["opacities"])

    def get_quats(self) -> torch.Tensor:
        return self.quat_activation(self.params["quats"])

    def get_coefs(self) -> torch.Tensor:
        assert "motion_coefs" in self.params
        return self.motion_coef_activation(self.params["motion_coefs"])

    def densify_params(self, should_split, should_dup):
        """
        densify gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_dup = x[should_dup]
            x_split = x[should_split].repeat([2] + [1] * (x.ndim - 1))
            if name == "scales":
                x_split -= math.log(1.6)
            x_new = nn.Parameter(torch.cat([x[~should_split], x_dup, x_split], dim=0))
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def cull_params(self, should_cull):
        """
        cull gaussians
        """
        updated_params = {}
        for name, x in self.params.items():
            x_new = nn.Parameter(x[~should_cull])
            updated_params[name] = x_new
            self.params[name] = x_new
        return updated_params

    def reset_opacities(self, new_val):
        """
        reset all opacities to new_val
        """
        self.params["opacities"].data.fill_(new_val)
        updated_params = {"opacities": self.params["opacities"]}
        return updated_params


class MotionBases(nn.Module):
    def __init__(self, rots, transls):
        super().__init__()
        self.num_frames = rots.shape[1]
        self.num_bases = rots.shape[0]
        assert check_bases_sizes(rots, transls)
        self.params = nn.ParameterDict(
            {
                "rots": nn.Parameter(rots),
                "transls": nn.Parameter(transls),
            }
        )

    @staticmethod
    def init_from_state_dict(state_dict, prefix="params."):
        param_keys = ["rots", "transls"]
        assert all(f"{prefix}{k}" in state_dict for k in param_keys)
        args = {k: state_dict[f"{prefix}{k}"] for k in param_keys}
        return MotionBases(**args)


    def compute_transforms(self, ts: torch.Tensor, coefs: torch.Tensor) -> torch.Tensor:
        """
        :param ts (B)
        :param coefs (G, K)
        returns transforms (G, B, 3, 4)
        """
        # Local Time interpolation in Points-based Motion
        # ts (num_fg, B)
        if len(ts.shape) == 1:
            ts = ts.unsqueeze(0)
        ts_pre = torch.floor(ts.clone()).clamp(0., self.params["transls"].shape[1]-1).int()
        ts_next = torch.ceil(ts.clone()).clamp(0., self.params["transls"].shape[1]-1).int()

        #              (20, 8, 3)             ()
        #  the same pre and next for all gaussians
        transls_pre = self.params["transls"][:, ts_pre[0, :]]
        rots_pre = self.params["rots"][:, ts_pre[0, :]]

        transls_next = self.params["transls"][:, ts_next[0, :]]
        rots_next = self.params["rots"][:, ts_next[0, :]]

        transls_pre = torch.einsum("pk,kni->pni", coefs, transls_pre)  # (G, B, 3)  torch.Size([22200, 1, 3])
        rots_pre = torch.einsum("pk,kni->pni", coefs, rots_pre)  # (G, B, 6)  torch.Size([22200, 1, 6])
        transls_next = torch.einsum("pk,kni->pni", coefs, transls_next)  # (G, B, 3)  torch.Size([22200, 1, 3])
        rots_next = torch.einsum("pk,kni->pni", coefs, rots_next)  # (G, B, 6)  torch.Size([22200, 1, 6])
        
        num_fg = transls_pre.shape[0]
        if len(ts.shape) == 2 and ts.shape[0] == 1:  # (G, B, 1)
            ts = ts.repeat(num_fg, 1)
            ts_pre = ts_pre.repeat(num_fg, 1)
            ts_next = ts_next.repeat(num_fg, 1)
        w = (ts - ts_pre)  # next motion的权重
        w = w.unsqueeze(-1)
        # print(ts.shape, w.shape, transls_pre.shape, transls_next.shape)
        transls = (1. - w) * transls_pre + w * transls_next
        rots = (1. - w) * rots_pre+ w * rots_next
        rotmats = cont_6d_to_rmat(rots)  # (K, B, 3, 3)
        # print(rotmats.shape, transls[..., None].shape)
        return torch.cat([rotmats, transls[..., None]], dim=-1)


def check_gaussian_sizes(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    motion_coefs: torch.Tensor | None = None,
) -> bool:
    dims = means.shape[:-1]
    leading_dims_match = (
        quats.shape[:-1] == dims
        and scales.shape[:-1] == dims
        and colors.shape[:-1] == dims
        and opacities.shape == dims
    )
    if motion_coefs is not None and motion_coefs.numel() > 0:
        leading_dims_match &= motion_coefs.shape[:-1] == dims
    dims_correct = (
        means.shape[-1] == 3
        and (quats.shape[-1] == 4)
        and (scales.shape[-1] == 3)
        and (colors.shape[-1] == 3)
    )
    return leading_dims_match and dims_correct


def check_bases_sizes(motion_rots: torch.Tensor, motion_transls: torch.Tensor) -> bool:
    return (
        motion_rots.shape[-1] == 6
        and motion_transls.shape[-1] == 3
        and motion_rots.shape[:-2] == motion_transls.shape[:-2]
    )
