import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization
from torch import Tensor

from flow3d.params import GaussianParams, MotionBases
from flow3d.models.move_model import MoveModel
import os
import cv2

class SceneModel(nn.Module):
    def __init__(
        self,
        Ks: Tensor,
        w2cs: Tensor,
        fg_params: GaussianParams,
        motion_bases: MotionBases,
        bg_params: GaussianParams | None = None,
    ):
        super().__init__()
        self.num_frames = motion_bases.num_frames
        self.fg = fg_params
        self.motion_bases = motion_bases
        self.bg = bg_params
        scene_scale = 1.0 if bg_params is None else bg_params.scene_scale
        self.register_buffer("bg_scene_scale", torch.as_tensor(scene_scale))
        self.register_buffer("Ks", Ks)
        self.register_buffer("w2cs", w2cs)

        self._current_xys = None
        self._current_radii = None
        self._current_img_wh = None

        self.move_model = MoveModel(num_fg=self.num_fg_gaussians, camera_mode='linear').cuda()

    @property
    def num_gaussians(self) -> int:
        return self.num_bg_gaussians + self.num_fg_gaussians

    @property
    def num_bg_gaussians(self) -> int:
        return self.bg.num_gaussians if self.bg is not None else 0

    @property
    def num_fg_gaussians(self) -> int:
        return self.fg.num_gaussians

    @property
    def num_motion_bases(self) -> int:
        return self.motion_bases.num_bases

    @property
    def has_bg(self) -> bool:
        return self.bg is not None

    def compute_poses_bg(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            means: (G, B, 3)
            quats: (G, B, 4)
        """
        assert self.bg is not None
        return self.bg.params["means"], self.bg.get_quats()

    def compute_transforms(
        self, ts: torch.Tensor, inds: torch.Tensor | None = None
    ) -> torch.Tensor:
        coefs = self.fg.get_coefs()  # (G, K)
        if inds is not None:
            coefs = coefs[inds]
        transfms = self.motion_bases.compute_transforms(ts, coefs)  # (G, B, 3, 4)
        return transfms

    def compute_poses_fg(
        self, ts: torch.Tensor | None, inds: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :returns means: (G, B, 3), quats: (G, B, 4)
        """
        means = self.fg.params["means"]  # (G, 3)
        quats = self.fg.get_quats()  # (G, 4)
        if inds is not None:
            means = means[inds]
            quats = quats[inds]
        if ts is not None:
            transfms = self.compute_transforms(ts, inds)  # (G, B, 3, 4)
            means = torch.einsum(
                "pnij,pj->pni",
                transfms,
                F.pad(means, (0, 1), value=1.0),
            )
            quats = roma.quat_xyzw_to_wxyz(
                (
                    roma.quat_product(
                        roma.rotmat_to_unitquat(transfms[..., :3, :3]),
                        roma.quat_wxyz_to_xyzw(quats[:, None]),
                    )
                )
            )
            quats = F.normalize(quats, p=2, dim=-1)
        else:
            means = means[:, None]
            quats = quats[:, None]
        return means, quats

    def compute_poses_all(
        self, ts: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        means, quats = self.compute_poses_fg(ts)     #  运动区域的Gaussian
        if self.has_bg:
            bg_means, bg_quats = self.compute_poses_bg()    # 背景区域的Gaussian
            means = torch.cat(
                [means, bg_means[:, None].expand(-1, means.shape[1], -1)], dim=0
            ).contiguous()
            quats = torch.cat(
                [quats, bg_quats[:, None].expand(-1, means.shape[1], -1)], dim=0
            ).contiguous()
        return means, quats

    def get_colors_all(self) -> torch.Tensor:
        colors = self.fg.get_colors()
        if self.bg is not None:
            colors = torch.cat([colors, self.bg.get_colors()], dim=0).contiguous()
        return colors

    def get_scales_all(self) -> torch.Tensor:
        scales = self.fg.get_scales()
        if self.bg is not None:
            scales = torch.cat([scales, self.bg.get_scales()], dim=0).contiguous()
        return scales

    def get_opacities_all(self) -> torch.Tensor:
        """
        :returns colors: (G, 3), scales: (G, 3), opacities: (G, 1)
        """
        opacities = self.fg.get_opacities()
        if self.bg is not None:
            opacities = torch.cat(
                [opacities, self.bg.get_opacities()], dim=0
            ).contiguous()
        return opacities

    @staticmethod
    def init_from_state_dict(state_dict, prefix=""):
        fg = GaussianParams.init_from_state_dict(
            state_dict, prefix=f"{prefix}fg.params."
        )
        bg = None
        if any("bg." in k for k in state_dict):
            bg = GaussianParams.init_from_state_dict(
                state_dict, prefix=f"{prefix}bg.params."
            )
        motion_bases = MotionBases.init_from_state_dict(
            state_dict, prefix=f"{prefix}motion_bases.params."
        )
        Ks = state_dict[f"{prefix}Ks"]
        w2cs = state_dict[f"{prefix}w2cs"]
        return SceneModel(Ks, w2cs, fg, motion_bases, bg)

    def render(
        self,
        # A single time instance for view rendering.
        t: int | None,
        w2cs: torch.Tensor,  # (C, 4, 4)
        Ks: torch.Tensor,  # (C, 3, 3)
        img_wh: tuple[int, int],
        # Multiple time instances for track rendering: (B,).
        target_ts: torch.Tensor | None = None,  # (B)
        target_w2cs: torch.Tensor | None = None,  # (B, 4, 4)
        bg_color: torch.Tensor | float = 1.0,
        colors_override: torch.Tensor | None = None,
        means: torch.Tensor | None = None,
        quats: torch.Tensor | None = None,
        target_means: torch.Tensor | None = None,
        return_color: bool = True,
        return_depth: bool = False,
        return_mask: bool = False,
        fg_only: bool = False,            
        bg_only: bool = False,  
        filter_mask: torch.Tensor | None = None,
        epoch=1,
        mode="mid",
        stage="second",
    ) -> dict:
        check_mode = mode
        
        assert (fg_only==False and bg_only == False) or (fg_only==True and bg_only == False) or (fg_only==False and bg_only == True)

        device = w2cs.device
        C = w2cs.shape[0]

        W, H = img_wh

        if fg_only==True and bg_only==False:
            N = self.num_fg_gaussians
            pose_fnc = self.compute_poses_fg
        elif fg_only==False and bg_only==True:
            N = self.num_bg_gaussians
        else:
            N = self.num_gaussians
            pose_fnc = self.compute_poses_all

        if colors_override is None:
            if return_color:
                if fg_only==True and bg_only==False:
                    colors_override = (self.fg.get_colors())
                elif fg_only==False and bg_only==True:
                    colors_override = (self.bg.get_colors())
                else:
                    colors_override = (self.get_colors_all())
            else:
                colors_override = torch.zeros(N, 0, device=device)

        D = colors_override.shape[-1]

        if fg_only==True and bg_only==False:
            scales = self.fg.get_scales()
            opacities = self.fg.get_opacities()
        elif fg_only==False and bg_only==True:
            scales = self.bg.get_scales()
            opacities = self.bg.get_opacities()
        else:
            scales = self.get_scales_all()
            opacities = self.get_opacities_all()

        if isinstance(bg_color, float):
            bg_color = torch.full((C, D), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        mode = "RGB"
        ds_expected = {"img": D}

        if return_mask:
            if fg_only==False and bg_only == False:
                mask_values = torch.zeros((self.num_gaussians, 1), device=device)
                mask_values[: self.num_fg_gaussians] = 1.0
            elif fg_only==True and bg_only == False:
                mask_values = torch.ones((self.num_fg_gaussians, 1), device=device)
            elif fg_only==False and bg_only == True:
                mask_values = torch.ones((self.num_bg_gaussians, 1), device=device) 

            colors_override = torch.cat([colors_override, mask_values], dim=-1)
            bg_color = torch.cat([bg_color, torch.zeros(C, 1, device=device)], dim=-1)
            ds_expected["mask"] = 1

        blur_num_cameras = 11 
        assert w2cs.shape[0] == 1
        RTs, times,  deltaT = self.move_model.forward_start_end_mid(info={'R':w2cs[0, :3, :3], 
                                                                          'T':w2cs[0, :3, 3:4], 
                                                                           'timestep':t},  
                                                                        num_cameras=blur_num_cameras, 
                                                                        mode='uniform',
                                                                        stage=stage,
                                                                        )

        B = 0
        if target_ts is not None:
            B = target_ts.shape[0]

            if target_means is None:
                target_means, _ = pose_fnc(target_ts)  # [G, B, 3]
            if target_w2cs is not None:
                target_w2cs[:, :3]
                target_w2cs_refined = []
                for b in range(0, B):
                    target_w2c_refined, _,  _ = self.move_model.forward_start_end_mid(info={'R':target_w2cs[b, :3, :3], 
                                                                                            'T':target_w2cs[b, :3, 3:4], 
                                                                                            'timestep':target_ts[b]},
                                                                                            num_cameras=3, 
                                                                                            mode='uniform',
                                                                                            stage=stage)
                    target_w2cs_refined.append(target_w2c_refined[1,:,:])
                target_w2cs_refined = torch.stack(target_w2cs_refined, 0)

                target_means = torch.einsum(
                    "bij,pbj->pbi",
                    target_w2cs[:, :3],
                    F.pad(target_means, (0, 1), value=1.0),
                )
            track_3d_vals = target_means.flatten(-2)  # (G, B * 3)
            d_track = track_3d_vals.shape[-1]
            colors_override = torch.cat([colors_override, track_3d_vals], dim=-1)
            bg_color = torch.cat(
                [bg_color, torch.zeros(C, track_3d_vals.shape[-1], device=device)],
                dim=-1,
            )
            ds_expected["tracks_3d"] = d_track

        assert colors_override.shape[-1] == sum(ds_expected.values())
        assert bg_color.shape[-1] == sum(ds_expected.values())

        if return_depth:
            mode = "RGB+ED"
            ds_expected["depth"] = 1

        if filter_mask is not None:
            assert filter_mask.shape == (N,)
            scales = scales[filter_mask]
            opacities = opacities[filter_mask]
            colors_override = colors_override[filter_mask]
        

        self.training = True
        if self.training:
            all_render_colors = []
            all_alphas = []
            all_info = []
            all_raddi = []
            all_means2s = []

            if check_mode == 'mid':
                RTs = RTs[blur_num_cameras//2:blur_num_cameras//2+1][:3, :3]
                times = times[:, blur_num_cameras//2:blur_num_cameras//2+1]
            if check_mode == 'start':
                RTs = RTs[0:1][:3, :3]
                times = times[:, 0:1] 
            if check_mode == 'end':
                RTs = RTs[blur_num_cameras-1:blur_num_cameras][:3, :3]
                times = times[:, blur_num_cameras-1:blur_num_cameras]

            for ii in range(0, len(RTs)):
                transR = RTs[ii][:3, :3]
                transT = RTs[ii][:3, 3:4]
                time = times[:, ii:ii+1]  

                if fg_only==True and bg_only==False:
                    means, quats = self.compute_poses_fg(
                        time if t is not None else None
                    )
                    means = means[:, 0]
                    quats = quats[:, 0]
                    N = self.num_fg_gaussians

                elif fg_only==False and bg_only==True:
                    means, quats = self.compute_poses_bg()
                    N = self.num_bg_gaussians

                else:
                    if self.num_fg_gaussians == 0:
                        means, quats = self.compute_poses_bg()
                        N = self.num_bg_gaussians
                    else:
                        means, quats = self.compute_poses_all(
                            time if t is not None else None
                        )
                        means = means[:, 0]
                        quats = quats[:, 0]
                        N = self.num_gaussians

                means = transR @ means.permute(1, 0) + transT
                means = means.permute(1, 0)

                if filter_mask is not None:
                    assert filter_mask.shape == (N,)
                    means = means[filter_mask]
                    quats = quats[filter_mask]

                render_colors, alphas, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors_override,
                    backgrounds=bg_color,
                    viewmats=w2cs,  # [C, 4, 4]
                    Ks=Ks,  # [C, 3, 3]
                    width=W,
                    height=H,
                    packed=False,
                    render_mode=mode,
                )

                tmp_dir = "/home/wrl/8T/DeblurNeRF/release/TCSVT/Rebuttal/Deblur4DGS/ckpts_low/tmp/2"
                os.makedirs(tmp_dir, exist_ok=True)

                cv2.imwrite(os.path.join(tmp_dir, str(t)+"_"+str(ii)+'.png'), render_colors[0, :, :, :3].detach().cpu().numpy()[:, :, ::-1]*255)
                
                all_render_colors.append(render_colors)
                all_alphas.append(alphas)
                all_info.append(info)
                all_raddi.append(info['radii'])
                all_means2s.append(info["means2d"])

            if check_mode in ['start', 'mid', 'end']:
                avg_render_colors = all_render_colors[0]
            else:
                avg_render_colors = torch.stack(all_render_colors, dim=0).mean(0)

            render_colors[:, :, :, 0:avg_render_colors.shape[-1]] = avg_render_colors[:, :, :, 0:avg_render_colors.shape[-1]]
            render_colors[:, :, :, 3:4] = torch.stack(all_render_colors, dim=0).max(0)[0][:, :, :, 3:4]
            render_colors[:, :, :, 16:17] = torch.stack(all_render_colors, dim=0).min(0)[0][:, :, :, 16:17]
            alphas = torch.stack(all_alphas, dim=0).mean(0)
            info = all_info[len(RTs)//2]

            pred_sharp_img = all_render_colors[len(RTs)//2][:, :, :, 0:3]
        else:
            if check_mode == 'mid':
                transR = RTs[blur_num_cameras//2][:3, :3]
                transT = RTs[blur_num_cameras//2][:3, 3:4]
                time = times[:, blur_num_cameras//2:blur_num_cameras//2+1]
            if check_mode == 'start':
                transR = RTs[0][:3, :3]
                transT = RTs[0][:3, 3:4]
                time = times[:, 0:1]   
            if check_mode == 'end':
                transR = RTs[blur_num_cameras - 1][:3, :3]
                transT = RTs[blur_num_cameras - 1][:3, 3:4]
                time = times[:, blur_num_cameras - 1:blur_num_cameras] 

            if fg_only==True and bg_only==False:
                means, quats = self.compute_poses_fg(
                    time if t is not None else None)
                means = means[:, 0]
                quats = quats[:, 0]
                N = self.num_fg_gaussians
            elif fg_only==False and bg_only==True:
                N = self.num_bg_gaussians
            else:
                if self.fg.num_gaussians == 0:
                    means, quats = self.compute_poses_bg()
                    N = self.num_bg_gaussians
                else:
                    means, quats = self.compute_poses_all(
                        time if t is not None else None
                    )
                    means = means[:, 0]
                    quats = quats[:, 0]
                    N = self.num_gaussians

            means = transR @ means.permute(1, 0) + transT
            means = means.permute(1, 0)

            if filter_mask is not None:
                assert filter_mask.shape == (N,)
                means = means[filter_mask]
                quats = quats[filter_mask]

            render_colors, alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors_override,
                backgrounds=bg_color,
                viewmats=w2cs,  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=W,
                height=H,
                packed=False,
                render_mode=mode,
            )
        
        # Populate the current data for adaptive gaussian control.
        if self.training and info["means2d"].requires_grad:
            for info in all_info:
                info["means2d"].retain_grad() 
            self._current_xys = [info["means2d"] for info in all_info]
            self._current_radii = [info["radii"] for info in all_info]
            self._current_img_wh = img_wh
            # We want to be able to access to xys' gradients later in a
            # torch.no_grad context.
            # self._current_xys.retain_grad()

        assert render_colors.shape[-1] == sum(ds_expected.values())
        outputs = torch.split(render_colors, list(ds_expected.values()), dim=-1)

        out_dict = {}
        for i, (name, dim) in enumerate(ds_expected.items()):
            x = outputs[i]
            assert x.shape[-1] == dim, f"{x.shape[-1]=} != {dim=}"
            if name == "tracks_3d":
                x = x.reshape(C, H, W, B, 3)
            out_dict[name] = x
        out_dict["acc"] = alphas

        out_dict['deltaT'] = deltaT.unsqueeze(0)
        out_dict['RTs'] = RTs
        if self.training and target_ts is not None:
            out_dict['pred_sharp_img'] = pred_sharp_img
            out_dict['exposure_imgs'] = torch.stack(all_render_colors, 0)



        out_dict['exposure_imgs'] = torch.stack(all_render_colors, 0)
        return out_dict