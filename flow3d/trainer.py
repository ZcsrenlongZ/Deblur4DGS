import functools
import time
from dataclasses import asdict
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from pytorch_msssim import SSIM
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.loss_utils import (
    compute_gradient_loss,
    compute_se3_smoothness_loss,
    compute_z_acc_loss,
    masked_l1_loss,
    AlignedLoss,
    VGGLoss,
    TVLoss
)
from flow3d.metrics import PCK, mLPIPS, mPSNR, mSSIM
from flow3d.scene_model import SceneModel
from flow3d.vis.utils import get_server
from flow3d.vis.viewer import DynamicViewer
import torch.nn as nn

from flow3d.models.move_model import MoveModel
import os
import cv2


class Trainer:
    def __init__(
        self,
        model: SceneModel,
        device: torch.device,
        lr_cfg: SceneLRConfig,
        losses_cfg: LossesConfig,
        optim_cfg: OptimizerConfig,
        # Logging.
        work_dir: str,
        port: int | None = None,
        log_every: int = 10,
        checkpoint_every: int = 200,
        validate_every: int = 500,
        validate_video_every: int = 1000,
        validate_viewer_assets_every: int = 100,
    ):
        self.device = device
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.validate_every = validate_every
        self.validate_video_every = validate_video_every
        self.validate_viewer_assets_every = validate_viewer_assets_every

        self.model = model
        self.num_frames = model.num_frames

        self.lr_cfg = lr_cfg
        self.losses_cfg = losses_cfg
        self.optim_cfg = optim_cfg

        self.reset_opacity_every = (
            self.optim_cfg.reset_opacity_every_n_controls * self.optim_cfg.control_every
        )
        self.optimizers, self.scheduler = self.configure_optimizers()

        # running stats for adaptive density control
        self.running_stats = {
            "xys_grad_norm_acc": torch.zeros(self.model.num_gaussians, device=device),
            "vis_count": torch.zeros(
                self.model.num_gaussians, device=device, dtype=torch.int64
            ),
            "max_radii": torch.zeros(self.model.num_gaussians, device=device),
        }

        self.work_dir = work_dir
        self.writer = SummaryWriter(log_dir=work_dir)
        self.global_step = 0
        self.epoch = 0

        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            self.viewer = DynamicViewer(
                server, self.render_fn, model.num_frames, work_dir, mode="training"
            )

        # metrics
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.psnr_metric = mPSNR()
        self.ssim_metric = mSSIM()
        self.lpips_metric = mLPIPS()
        self.pck_metric = PCK()
        self.bg_psnr_metric = mPSNR()
        self.fg_psnr_metric = mPSNR()
        self.bg_ssim_metric = mSSIM()
        self.fg_ssim_metric = mSSIM()
        self.bg_lpips_metric = mLPIPS()
        self.fg_lpips_metric = mLPIPS()

        self.pose_model_optimizer = torch.optim.Adam(
            [{"params": self.model.move_model.RT_main.parameters(), "lr":5e-4},
             {"params": self.model.move_model.RT_head0.parameters(), "lr":5e-4},
             {"params": self.model.move_model.RT_head1.parameters(), "lr":5e-4}])
        
        self.pose_model_optimizer.zero_grad(set_to_none=True)
        self.pose_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.pose_model_optimizer, T_max=24*500, eta_min=1e-5)
        
        self.time_model_optimizer = torch.optim.Adam(
            [{"params": self.model.move_model.time_params, "lr":1e-1}])
        self.time_model_optimizer.zero_grad(set_to_none=True)
        self.time_model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.time_model_optimizer, T_max=24*200, eta_min=1e-5)


        self.alignloss = AlignedLoss().cuda()
        self.max_pool = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
 

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def save_checkpoint(self, path: str):
        model_dict = self.model.state_dict()
        move_model_dict = self.model.move_model.state_dict()
        optimizer_dict = {k: v.state_dict() for k, v in self.optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in self.scheduler.items()}
        ckpt = {
            "model": model_dict,
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "move_model": move_model_dict
        }
        torch.save(ckpt, path)
        guru.info(f"Saved checkpoint at {self.global_step=} to {path}")

    @staticmethod
    def init_from_checkpoint( 
        path: str, device: torch.device, *args, **kwargs
    ) -> tuple["Trainer", int]:
        guru.info(f"Loading checkpoint from {path}")
        ckpt = torch.load(path)
        state_dict = ckpt["model"]
        model = SceneModel.init_from_state_dict(state_dict)
        model = model.to(device)

        if "move_model" in ckpt.keys():
            num_fg = model.num_fg_gaussians
            move_model = MoveModel(num_fg, camera_mode='linear').cuda()
            move_model_dict = ckpt["move_model"]
            if move_model_dict['time_params'].shape[0] != num_fg:
                move_model_dict.pop('time_params')
            move_model.load_state_dict(move_model_dict, strict=False)
            move_model = move_model.to(device)
            model.move_model = move_model

        trainer = Trainer(model, device, *args, **kwargs)
        if "optimizers" in ckpt:
            trainer.load_checkpoint_optimizers(ckpt["optimizers"])
        if "schedulers" in ckpt:
            trainer.load_checkpoint_schedulers(ckpt["schedulers"])
        trainer.global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        trainer.set_epoch(start_epoch)
        return trainer, start_epoch

    def load_checkpoint_optimizers(self, opt_ckpt):
        for k, v in self.optimizers.items():
            v.load_state_dict(opt_ckpt[k])

    def load_checkpoint_schedulers(self, sched_ckpt):
        for k, v in self.scheduler.items():
            v.load_state_dict(sched_ckpt[k])

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = 0
        if self.viewer is not None:
            t = (
                int(self.viewer._playback_guis[0].value)
                if not self.viewer._canonical_checkbox.value
                else None
            )
        self.model.training = False
        img = self.model.render(t, w2c[None], K[None], img_wh)["img"][0]
        return (img.cpu().numpy() * 255.0).astype(np.uint8)
    
    def train_step(self, batch1, batch2, batch3, epoch, dyn_time_ids, stage, batch4=None):
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.1)
            self.viewer.lock.acquire()

        loss_1 = 0.
        if batch1 is not None:
            loss_1, stats, num_rays_per_step, num_rays_per_sec  = self.compute_static_losses(batch1, epoch, stage=stage)
        
        loss_2 = 0.
        if batch2 is not None:
            if batch4 is None:
                loss_2, stats, _, _ = self.compute_dynamic_losses(batch2[0], epoch, dyn_time_ids=dyn_time_ids, stage=stage, batch4=None)
            else:
                loss_2, stats, _, _ = self.compute_dynamic_losses(batch2[0], epoch, dyn_time_ids=dyn_time_ids, stage=stage, batch4=batch4[0])
        
        loss_3 = 0.
        if batch3 is not None:
            loss_3, stats, num_rays_per_step, num_rays_per_sec  = self.compute_static_reg_losses(batch3, epoch, stage=stage)
        
        loss =  loss_1 + loss_2 + loss_3
        
        if loss.isnan():
            guru.info(f"Loss is NaN at step {self.global_step}!!")
            import ipdb

            ipdb.set_trace()
        loss.backward()

        for module in self.optimizers.keys():
            opt = self.optimizers[module]
            sched = self.scheduler[module]
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

    
        if stage == "first": 
            if epoch > 20 and self.global_step % 25 ==0:
                self.pose_model_optimizer.step()
                self.pose_model_optimizer.zero_grad(set_to_none=True)
            self.pose_model_scheduler.step()
        elif stage == "second":
            if epoch > 20 and self.global_step % 25 ==0:
                self.pose_model_optimizer.step()
                self.pose_model_optimizer.zero_grad(set_to_none=True)
            self.pose_model_scheduler.step()

            if self.global_step % 25 ==0:
                self.time_model_optimizer.step()
                self.time_model_optimizer.zero_grad(set_to_none=True)
            self.time_model_scheduler.step()

        self.log_dict(stats)
        self.global_step += 1

        if batch1 is None:
            self.run_control_steps(only_fg=True)
        else:
            self.run_control_steps(only_fg=False)

        if self.viewer is not None:
            self.viewer.lock.release()
            self.viewer.state.num_train_rays_per_sec = num_rays_per_sec
            if self.viewer.mode == "training":
                self.viewer.update(self.global_step, num_rays_per_step)

        if self.global_step % self.checkpoint_every == 0:
            self.save_checkpoint(f"{self.work_dir}/checkpoints/last.ckpt")

        return loss.item()
    
    def compute_static_losses(self, batch_sta, epoch=1, stage="first"):
        self.model.training = True
        loss_batch = []
        all_RTS = []
        for batch in batch_sta:
            B = batch["imgs"].shape[0]
            W, H = img_wh = batch["imgs"].shape[2:0:-1]
            N = batch["target_ts"][0].shape[0]

            # (B,).
            ts = batch["ts"]
            # (B, 4, 4).
            w2cs = batch["w2cs"]
            # (B, 3, 3).
            Ks = batch["Ks"]
            # (B, H, W, 3).
            imgs = batch["imgs"]
            # (B, H, W).
            valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
            # (B, H, W).
            masks = batch["masks"]
            masks *= valid_masks
            # (B, H, W).
            depths = batch["depths"]

            _tic = time.time()
            # (B, G, 3).
            means, quats = self.model.compute_poses_bg()  # (G, B, 3), (G, B, 4)
            device = means.device
            means = means.transpose(0, 1)
            quats = quats.transpose(0, 1)
            # [(N, G, 3), ...].

            loss = 0.0

            bg_colors = []
            rendered_all = []
            self._batched_xys = []
            self._batched_radii = []
            self._batched_img_wh = []

            # Batch渲染
            for i in range(B):
                bg_color = torch.ones(1, 3, device=device)
                rendered = self.model.render(
                    ts[i].item(),
                    w2cs[None, i],
                    Ks[None, i],
                    img_wh,
                    target_ts=None,
                    target_w2cs=None,
                    bg_color=bg_color,
                    means=None,
                    quats=None,
                    target_means=None,
                    return_depth=True,
                    return_mask=self.model.has_bg,
                    fg_only=False,
                    bg_only=True,
                    epoch=epoch,
                    mode='blury',
                    stage=stage,
                )
                rendered_all.append(rendered)
                bg_colors.append(bg_color)
                if (
                    self.model._current_xys is not None
                    and self.model._current_radii is not None
                    and self.model._current_img_wh is not None
                ):
                    self._batched_xys.append(self.model._current_xys) 
                    self._batched_radii.append(self.model._current_radii)
                    self._batched_img_wh.append(self.model._current_img_wh)

            # Necessary to make viewer work.
            num_rays_per_step = H * W * B
            num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

            # (B, H, W, N, *).
            rendered_all = {
                key: (
                    torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                    if rendered_all[0][key] is not None
                    else None
                )
                for key in rendered_all[0]
            }
            bg_colors = torch.cat(bg_colors, dim=0)

            all_RTS.append(rendered_all['RTs'])

            # Compute losses.
            # (B * N).
            if not self.model.has_bg:
                imgs = (
                    imgs * masks[..., None]
                    + (1.0 - masks[..., None]) * bg_colors[:, None, None]
                )
            else:
                imgs = (
                    imgs * valid_masks[..., None]
                    + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
                )

            # RGB loss.
            rendered_imgs = cast(torch.Tensor, rendered_all["img"])
            if self.model.has_bg:
                rendered_imgs = (
                    rendered_imgs * valid_masks[..., None]
                    + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
                )

            mask_dilated = self.max_pool(masks.unsqueeze(0)).permute(0, 2, 3, 1)
            rgb_loss = 0.8 * F.l1_loss(rendered_imgs*(1. - mask_dilated), imgs*(1. - mask_dilated)) + 0.2 * (
                1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2)*(1. - mask_dilated.permute(0, 3, 1, 2)), 
                            imgs.permute(0, 3, 1, 2)*(1. - mask_dilated.permute(0, 3, 1, 2)))
            )
            
            loss += rgb_loss * self.losses_cfg.w_rgb


            depth_masks = (1. - mask_dilated.float())

            pred_depth = cast(torch.Tensor, rendered_all["depth"])
            pred_disp = 1.0 / (pred_depth + 1e-5)
            tgt_disp = 1.0 / (depths[..., None] + 1e-5)
            depth_loss = masked_l1_loss(
                pred_disp,
                tgt_disp,
                mask=depth_masks,
                quantile=0.98,
            )
            loss += depth_loss * self.losses_cfg.w_depth_reg

            depth_gradient_loss = compute_gradient_loss(
                pred_disp,
                tgt_disp,
                mask=depth_masks > 0.5,
                quantile=0.95,
            )
            loss += depth_gradient_loss * self.losses_cfg.w_depth_grad

            loss += (
                self.losses_cfg.w_scale_var
                * torch.var(self.model.bg.params["scales"], dim=-1).mean()
            )

            loss_batch.append(loss)
            stats = {
                "train/static_rgb_loss": rgb_loss.item(),
                "train/num_bg_gaussians": self.model.num_bg_gaussians,
            }

        # continuous pose
        if len(batch_sta) == 3:
            loss = torch.mean(torch.stack(loss_batch, 0))
            reg = torch.mean(torch.abs(all_RTS[0][-1] - all_RTS[1][0])) + torch.mean(torch.abs(all_RTS[2][0] - all_RTS[1][-1]))
            loss + reg
        return loss, stats, num_rays_per_step, num_rays_per_sec

    def compute_dynamic_losses(self, batch, epoch=1, dyn_time_ids=None, stage="second", batch4=None):
        self.model.training = True
        B = batch["imgs"].shape[0]
        W, H = img_wh = batch["imgs"].shape[2:0:-1]
        N = batch["target_ts"][0].shape[0]

        # (B,).
        ts = batch["ts"]
        # (B, 4, 4).
        w2cs = batch["w2cs"]
        # (B, 3, 3).
        Ks = batch["Ks"]
        # (B, H, W, 3).
        imgs = batch["imgs"]
        # (B, H, W).
        valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
        # (B, H, W).
        masks = batch["masks"]
        masks *= valid_masks
        # (B, H, W).
        depths = batch["depths"]
        # [(P, 2), ...].
        query_tracks_2d = batch["query_tracks_2d"]
        # [(N,), ...].
        target_ts = batch["target_ts"]
        # [(N, 4, 4), ...].
        target_w2cs = batch["target_w2cs"]
        # [(N, 3, 3), ...].
        target_Ks = batch["target_Ks"]
        # [(N, P, 2), ...].
        target_tracks_2d = batch["target_tracks_2d"]
        # [(N, P), ...].
        target_visibles = batch["target_visibles"]
        # [(N, P), ...].
        target_invisibles = batch["target_invisibles"]
        # [(N, P), ...].
        target_confidences = batch["target_confidences"]
        # [(N, P), ...].
        target_track_depths = batch["target_track_depths"]

        _tic = time.time()
        # (B, G, 3).
        means, quats = self.model.compute_poses_all(ts)  # (G, B, 3), (G, B, 4)
        device = means.device
        means = means.transpose(0, 1)
        quats = quats.transpose(0, 1)
        # [(N, G, 3), ...].
        target_ts_vec = torch.cat(target_ts)
        # (B * N, G, 3).
        target_means, _ = self.model.compute_poses_all(target_ts_vec)
        target_means = target_means.transpose(0, 1)
        target_mean_list = target_means.split(N)
        # num_frames = self.model.num_frames
        num_frames = dyn_time_ids.shape[0]

        loss = 0.0

        bg_colors = []
        rendered_all = []
        self._batched_xys = []
        self._batched_radii = []
        self._batched_img_wh = []

        for i in range(B):
            bg_color = torch.ones(1, 3, device=device)
            rendered = self.model.render(
                ts[i].item(),
                w2cs[None, i],
                Ks[None, i],
                img_wh,
                target_ts=target_ts[i],
                target_w2cs=target_w2cs[i],
                bg_color=bg_color,
                means=means[i],
                quats=quats[i],
                target_means=target_mean_list[i].transpose(0, 1),
                return_depth=True,
                return_mask=self.model.has_bg,
                fg_only=False,
                bg_only=False,
                epoch=epoch,
                mode='blury',
                stage=stage
            )
            rendered_all.append(rendered)
            bg_colors.append(bg_color)

            if (
                self.model._current_xys is not None
                and self.model._current_radii is not None
                and self.model._current_img_wh is not None
            ):
                self._batched_xys.append(self.model._current_xys)
                self._batched_radii.append(self.model._current_radii)
                self._batched_img_wh.append(self.model._current_img_wh)

        # Necessary to make viewer work.
        num_rays_per_step = H * W * B
        num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

        # (B, H, W, N, *).
        rendered_all = {
            key: (
                torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                if rendered_all[0][key] is not None
                else None
            )
            for key in rendered_all[0]
        }
        bg_colors = torch.cat(bg_colors, dim=0)

        # Compute losses.
        # (B * N).
        frame_intervals = (ts.repeat_interleave(N) - target_ts_vec).abs()
        if not self.model.has_bg:
            imgs = (
                imgs * masks[..., None]
                + (1.0 - masks[..., None]) * bg_colors[:, None, None]
            )
        else:
            imgs = (
                imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )
        # (P_all, 2).
        tracks_2d = torch.cat([x.reshape(-1, 2) for x in target_tracks_2d], dim=0)
        # (P_all,)
        visibles = torch.cat([x.reshape(-1) for x in target_visibles], dim=0)
        # (P_all,)
        confidences = torch.cat([x.reshape(-1) for x in target_confidences], dim=0)

        # RGB loss.
        rendered_imgs = cast(torch.Tensor, rendered_all["img"])
        if self.model.has_bg:
            rendered_imgs = (
                rendered_imgs * valid_masks[..., None]
                + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
            )

        mask_dilated = self.max_pool(masks.unsqueeze(0)).permute(0, 2, 3, 1)
        rgb_dyn_loss = 0.8 * F.l1_loss(rendered_imgs * mask_dilated, imgs * mask_dilated) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2) * mask_dilated.permute(0, 3, 1, 2), 
                          imgs.permute(0, 3, 1, 2) * mask_dilated.permute(0, 3, 1, 2))
        )
        loss += rgb_dyn_loss * self.losses_cfg.w_rgb 

        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, imgs) + 0.2 * (
            1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), 
                          imgs.permute(0, 3, 1, 2))
        )
        loss += rgb_loss * self.losses_cfg.w_rgb


        # all_render_colors = rendered_all['exposure_imgs']
        # if epoch % 50 == 0 or epoch == 199:
        #     save_dir = os.path.join(self.work_dir, 'vis', str(epoch))
        #     os.makedirs(save_dir, exist_ok=True)
        #     for jj in range(0, len(all_render_colors)):
        #         cv2.imwrite(os.path.join(save_dir, str(ts)+"_"+str(jj)+'.png'), all_render_colors[jj][0, :, :, :3].detach().cpu().numpy()[:, :, ::-1]*255)




        if epoch > 20:
            loss_cons = 0. 
            all_imgs = rendered_all['exposure_imgs']
            for ee in range(0, len(all_imgs)-1):
                img = all_imgs[ee:ee+1][:, 0, :, :, 0:3]
                img_next = all_imgs[ee+1:ee+2][:, 0, :, :, 0:3]
                mask = all_imgs[ee+1:ee+2][:, 0, :, :, 3:4].detach()
                loss_cons += self.alignloss(img.permute(0, 3, 1, 2), 
                                            img_next.permute(0, 3, 1, 2),
                                            mask=mask.permute(0, 3, 1, 2))

            for ee in range(1, len(all_imgs)):
                img = all_imgs[ee:ee+1][:, 0, :, :, 0:3]
                img_first = all_imgs[0:1][:, 0, :, :, 0:3]
                mask = all_imgs[0:1][:, 0, :, :, 3:4].detach()
                loss_cons += self.alignloss(img.permute(0, 3, 1, 2), 
                                            img_first.permute(0, 3, 1, 2).detach(),
                                            mask=mask.permute(0, 3, 1, 2))
            loss_cons = loss_cons / (len(all_imgs)-1)
            loss += loss_cons * 2.

        # Mask loss.
        if not self.model.has_bg:
            mask_loss = F.mse_loss(rendered_all["acc"], masks[..., None])  # type: ignore
        else:
            mask_loss = F.mse_loss(
                rendered_all["acc"], torch.ones_like(rendered_all["acc"])  # type: ignore
            ) + masked_l1_loss(
                rendered_all["mask"],
                masks[..., None],
                quantile=0.98,  # type: ignore
            )
        loss += mask_loss * self.losses_cfg.w_mask

        # (B * N, H * W, 3).
        pred_tracks_3d = (
            rendered_all["tracks_3d"].permute(0, 3, 1, 2, 4).reshape(-1, H * W, 3)  # type: ignore
        )
        pred_tracks_2d = torch.einsum(
            "bij,bpj->bpi", torch.cat(target_Ks), pred_tracks_3d
        )
        # (B * N, H * W, 1).
        mapped_depth = torch.clamp(pred_tracks_2d[..., 2:], min=1e-6)
        # (B * N, H * W, 2).
        pred_tracks_2d = pred_tracks_2d[..., :2] / mapped_depth

        # (B * N).
        w_interval = torch.exp(-2 * frame_intervals / num_frames)
        track_weights = confidences[..., None] * w_interval

        # (B, H, W).
        masks_flatten = torch.zeros_like(masks)
        for i in range(B):
            # This takes advantage of the fact that the query 2D tracks are
            # always on the grid.
            query_pixels = query_tracks_2d[i].to(torch.int64)
            masks_flatten[i, query_pixels[:, 1], query_pixels[:, 0]] = 1.0
        # (B * N, H * W).
        masks_flatten = (
            masks_flatten.reshape(-1, H * W).tile(1, N).reshape(-1, H * W) > 0.5
        )  

        track_2d_loss = masked_l1_loss(
            pred_tracks_2d[masks_flatten][visibles],        
            tracks_2d[visibles],                         
            mask=track_weights[visibles],                               
            quantile=0.98,
        ) / max(H, W)
        loss += track_2d_loss * self.losses_cfg.w_track

        depth_masks = (masks[..., None])
        pred_depth = cast(torch.Tensor, rendered_all["depth"])
        pred_disp = 1.0 / (pred_depth + 1e-5)
        tgt_disp = 1.0 / (depths[..., None] + 1e-5)
        depth_loss = masked_l1_loss(
            pred_disp,
            tgt_disp,
            mask=depth_masks,
            quantile=0.98,
        )
        loss += depth_loss * self.losses_cfg.w_depth_reg

        # mapped depth loss (using cached depth with EMA)
        mapped_depth_gt = torch.cat([x.reshape(-1) for x in target_track_depths], dim=0)
        mapped_depth_loss = masked_l1_loss(
            1 / (mapped_depth[masks_flatten][visibles] + 1e-5),  # P 1
            1 / (mapped_depth_gt[visibles, None] + 1e-5),        # P 1
            track_weights[visibles],                             # P 4
        )

        loss += mapped_depth_loss * self.losses_cfg.w_depth_const

        # bases should be smooth.
        small_accel_loss = compute_se3_smoothness_loss(
            self.model.motion_bases.params["rots"],
            self.model.motion_bases.params["transls"],
        )
        loss += small_accel_loss * self.losses_cfg.w_smooth_bases

        # tracks should be smooth
        ts = torch.clamp(ts, min=1, max=num_frames - 2)
        ts_neighbors = torch.cat((ts - 1, ts, ts + 1))
        transfms_nbs = self.model.compute_transforms(ts_neighbors)  # (G, 3n, 3, 4)
        means_fg_nbs = torch.einsum(
            "pnij,pj->pni",
            transfms_nbs,
            F.pad(self.model.fg.params["means"], (0, 1), value=1.0),
        )

        means_fg_nbs = means_fg_nbs.reshape(
            means_fg_nbs.shape[0], 3, -1, 3
        )  # [G, 3, n, 3]
        if self.losses_cfg.w_smooth_tracks > 0:
            small_accel_loss_tracks = 0.5 * (
                (2 * means_fg_nbs[:, 1:-1] - means_fg_nbs[:, :-2] - means_fg_nbs[:, 2:])
                .norm(dim=-1)
                .mean()
            )
            loss += small_accel_loss_tracks * self.losses_cfg.w_smooth_tracks

        # Constrain the std of scales.
        # TODO: do we want to penalize before or after exp?
        loss += (
            self.losses_cfg.w_scale_var
            * torch.var(self.model.fg.params["scales"], dim=-1).mean()
        )

        # Acceleration along ray direction should be small.
        z_accel_loss = compute_z_acc_loss(means_fg_nbs, w2cs)
        loss += self.losses_cfg.w_z_accel * z_accel_loss

        deltaT = rendered_all["deltaT"][0, :, 0]
        lambda_min = 0.5
        lambda_max = 0.75
        loss_reg = torch.max(torch.zeros_like(deltaT), lambda_min-deltaT) + torch.max(torch.zeros_like(deltaT), deltaT-lambda_max)
        loss += torch.mean(loss_reg) * 0.1

        if batch4 is None:
            masks_cur = masks.unsqueeze(0)
            masks_cur = F.interpolate(masks_cur, scale_factor=0.25, mode='area')
            masks_cur = masks_cur.permute(0, 2, 3, 1)
            render_dyn_down = rendered_all['pred_sharp_img']
            render_dyn_down = F.interpolate(render_dyn_down.permute(0, 3, 1, 2), scale_factor=0.25, mode='area')
            render_dyn_down = render_dyn_down.permute(0, 2, 3, 1) * masks_cur
            blury_dyn_down = F.interpolate(imgs.permute(0, 3, 1, 2), scale_factor=0.25, mode='area')
            blury_dyn_down = blury_dyn_down.permute(0, 2, 3, 1)
            blury_dyn_down = blury_dyn_down*masks_cur
            loss_keep = F.l1_loss(render_dyn_down, blury_dyn_down.detach())
            loss += loss_keep
        else:
            if epoch > 20:
                masks_cur = masks.unsqueeze(0)
                masks_cur = F.interpolate(masks_cur, scale_factor=0.25, mode='area')
                masks_cur = masks_cur.permute(0, 2, 3, 1)
                render_dyn_down = rendered_all['pred_sharp_img']
                render_dyn_down = F.interpolate(render_dyn_down.permute(0, 3, 1, 2), scale_factor=0.25, mode='area')
                render_dyn_down = render_dyn_down.permute(0, 2, 3, 1) * masks_cur

                img_keep = batch4["imgs"]
                img_keep = img_keep * masks_cur
                loss_keep = F.l1_loss(render_dyn_down, img_keep.detach())
                loss += loss_keep

        # Prepare stats for logging.
        stats = {
            "train/loss": loss.item(),
            "train/mask_loss": mask_loss.item(),
            "train/mapped_depth_loss": mapped_depth_loss.item(),
            "train/track_2d_loss": track_2d_loss.item(),
            "train/small_accel_loss": small_accel_loss.item(),
            "train/z_acc_loss": z_accel_loss.item(),
            "train/num_gaussians": self.model.num_gaussians,
            "train/num_fg_gaussians": self.model.num_fg_gaussians,
        }

        # Compute metrics.
        with torch.no_grad():
            psnr = self.psnr_metric(
                rendered_imgs, imgs, masks if not self.model.has_bg else valid_masks
            )
            self.psnr_metric.reset()
            stats["train/psnr"] = psnr
            fg_psnr = self.fg_psnr_metric(rendered_imgs, imgs, masks)
            self.fg_psnr_metric.reset()
            stats["train/fg_psnr"] = fg_psnr


        stats.update(
            **{
                "train/num_rays_per_sec": num_rays_per_sec,
                "train/num_rays_per_step": float(num_rays_per_step),
            }
        )

        return loss, stats, num_rays_per_step, num_rays_per_sec
    
    def compute_static_reg_losses(self, batch_sta, epoch=1, stage="second"):
        self.model.training = True
        loss_batch = []
        for batch in batch_sta:
            B = batch["imgs"].shape[0]
            W, H = img_wh = batch["imgs"].shape[2:0:-1]

            # (B,).
            ts = batch["ts"]
            # (B, 4, 4).
            w2cs = batch["w2cs"]
            # (B, 3, 3).
            Ks = batch["Ks"]
            # (B, H, W, 3).
            imgs = batch["imgs"]
            # (B, H, W).
            valid_masks = batch.get("valid_masks", torch.ones_like(batch["imgs"][..., 0]))
            # (B, H, W).
            masks = batch["masks"]
            masks *= valid_masks

            _tic = time.time()
            # (B, G, 3).
            means, quats = self.model.compute_poses_bg()  # (G, B, 3), (G, B, 4)
            device = means.device
            means = means.transpose(0, 1)
            quats = quats.transpose(0, 1)
            # [(N, G, 3), ...].

            loss = 0.0

            bg_colors = []
            rendered_all = []
            self._batched_xys = []
            self._batched_radii = []
            self._batched_img_wh = []

            for i in range(B):

                bg_color = torch.ones(1, 3, device=device)
                rendered = self.model.render(
                    ts[i].item(),
                    w2cs[None, i],
                    Ks[None, i],
                    img_wh,
                    target_ts=None,
                    target_w2cs=None,
                    bg_color=bg_color,
                    means=None,
                    quats=None,
                    target_means=None,
                    return_depth=True,
                    return_mask=self.model.has_bg,
                    fg_only=False,
                    bg_only=True,
                    epoch=epoch,
                    mode='mid', 
                    stage=stage,
                )
                rendered_all.append(rendered)
                bg_colors.append(bg_color)
                if (
                    self.model._current_xys is not None
                    and self.model._current_radii is not None
                    and self.model._current_img_wh is not None
                ):
                    self._batched_xys.append(self.model._current_xys) #  这个修改成了包含一条轨迹上的
                    self._batched_radii.append(self.model._current_radii)
                    self._batched_img_wh.append(self.model._current_img_wh)

            # Necessary to make viewer work.
            num_rays_per_step = H * W * B
            num_rays_per_sec = num_rays_per_step / (time.time() - _tic)

            # (B, H, W, N, *).
            rendered_all = {
                key: (
                    torch.cat([out_dict[key] for out_dict in rendered_all], dim=0)
                    if rendered_all[0][key] is not None
                    else None
                )
                for key in rendered_all[0]
            }
            bg_colors = torch.cat(bg_colors, dim=0)

            # Compute losses.
            # (B * N).
            if not self.model.has_bg:
                imgs = (
                    imgs * masks[..., None]
                    + (1.0 - masks[..., None]) * bg_colors[:, None, None]
                )
            else:
                imgs = (
                    imgs * valid_masks[..., None]
                    + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
                )

            # RGB loss.
            rendered_imgs = cast(torch.Tensor, rendered_all["img"])
            if self.model.has_bg:
                rendered_imgs = (
                    rendered_imgs * valid_masks[..., None]
                    + (1.0 - valid_masks[..., None]) * bg_colors[:, None, None]
                )

            mask_dilated = self.max_pool(masks.unsqueeze(0)).permute(0, 2, 3, 1)
            rgb_loss = 0.8 * F.l1_loss(rendered_imgs*(1. - mask_dilated), imgs*(1. - mask_dilated)) + 0.2 * (
                1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2)*(1. - mask_dilated.permute(0, 3, 1, 2)), 
                            imgs.permute(0, 3, 1, 2)*(1. - mask_dilated.permute(0, 3, 1, 2)))
            )
            
            loss += rgb_loss * self.losses_cfg.w_rgb
            loss += (
                self.losses_cfg.w_scale_var
                * torch.var(self.model.bg.params["scales"], dim=-1).mean()
            )

            loss_batch.append(loss)
            stats = {
                "train/static_rgb_loss": rgb_loss.item(),
                "train/num_bg_gaussians": self.model.num_bg_gaussians,
            }

        loss = torch.mean(torch.stack(loss_batch, 0))
        return loss, stats, num_rays_per_step, num_rays_per_sec

    def log_dict(self, stats: dict):
        for k, v in stats.items():
            self.writer.add_scalar(k, v, self.global_step)

    def run_control_steps(self, only_fg=False):

        global_step = self.global_step
        # Adaptive gaussian control.
        cfg = self.optim_cfg
        num_frames = self.model.num_frames
        ready = self._prepare_control_step()
        if (
            ready
            and global_step > cfg.warmup_steps
            and global_step % cfg.control_every == 0
            and global_step < cfg.stop_control_steps
        ):
            if (
                global_step < cfg.stop_densify_steps
                and global_step % self.reset_opacity_every > num_frames
            ):
                self._densify_control_step(global_step, only_fg=only_fg)
            if global_step % self.reset_opacity_every > min(3 * num_frames, 1000):
                self._cull_control_step(global_step, only_fg=only_fg)
            if global_step % self.reset_opacity_every == 0:
                self._reset_opacity_control_step(only_fg=only_fg)

            # Reset stats after every control.
            for k in self.running_stats:
                self.running_stats[k].zero_()

    @torch.no_grad()
    def _prepare_control_step(self) -> bool:
        # Prepare for adaptive gaussian control based on the current stats.
        if not (
            self.model._current_radii is not None
            and self.model._current_xys is not None
        ):
            guru.warning("Model not training, skipping control step preparation")
            return False

        batch_size = len(self._batched_xys)
        # these quantities are for each rendered view and have shapes (C, G, *)
        # must be aggregated over all views

        for _current_xys, _current_radii, _current_img_wh in zip(
            self._batched_xys, self._batched_radii, self._batched_img_wh
        ):
            assert len(_current_xys) == len(_current_radii)
            for ii in range(0, len(_current_xys)):
                sel = _current_radii[ii] > 0
                gidcs = torch.where(sel)[1]
                # normalize grads to [-1, 1] screen space
                xys_grad = _current_xys[ii].grad.clone()
                xys_grad[..., 0] *= _current_img_wh[0] / 2.0 * batch_size * len(_current_xys)
                xys_grad[..., 1] *= _current_img_wh[1] / 2.0 * batch_size * len(_current_xys)
                self.running_stats["xys_grad_norm_acc"].index_add_(
                    0, gidcs, xys_grad[sel].norm(dim=-1)
                )
                self.running_stats["vis_count"].index_add_(
                    0, gidcs, torch.ones_like(gidcs, dtype=torch.int64)
                )
                
                max_radii = torch.maximum(
                    self.running_stats["max_radii"].index_select(0, gidcs),
                    _current_radii[ii][sel] / max(_current_img_wh),
                )
                self.running_stats["max_radii"].index_put((gidcs,), max_radii)
        return True

    @torch.no_grad()
    def _densify_control_step(self, global_step, only_fg=False):
        assert (self.running_stats["vis_count"] > 0).any()

        cfg = self.optim_cfg
        xys_grad_avg = self.running_stats["xys_grad_norm_acc"] / self.running_stats[
            "vis_count"
        ].clamp_min(1)
        is_grad_too_high = xys_grad_avg > cfg.densify_xys_grad_threshold
        # Split gaussians.
        scales = self.model.get_scales_all()
        is_scale_too_big = scales.amax(dim=-1) > cfg.densify_scale_threshold
        if global_step < cfg.stop_control_by_screen_steps:
            is_radius_too_big = (
                self.running_stats["max_radii"] > cfg.densify_screen_threshold
            )
        else:
            is_radius_too_big = torch.zeros_like(is_grad_too_high, dtype=torch.bool)

        should_split = is_grad_too_high & (is_scale_too_big | is_radius_too_big)
        should_dup = is_grad_too_high & ~is_scale_too_big

        num_fg = self.model.num_fg_gaussians
        should_fg_split = should_split[:num_fg]
        num_fg_splits = int(should_fg_split.sum().item())
        should_fg_dup = should_dup[:num_fg]
        num_fg_dups = int(should_fg_dup.sum().item())

        should_bg_split = should_split[num_fg:]
        num_bg_splits = int(should_bg_split.sum().item())
        should_bg_dup = should_dup[num_fg:]
        num_bg_dups = int(should_bg_dup.sum().item())

        fg_param_map = self.model.fg.densify_params(should_fg_split, should_fg_dup)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            dup_in_optim(
                optimizer,
                [new_params],
                should_fg_split,
                num_fg_splits * 2 + num_fg_dups,
            )
        
        if not only_fg:
            if self.model.bg is not None:
                bg_param_map = self.model.bg.densify_params(should_bg_split, should_bg_dup)
                for param_name, new_params in bg_param_map.items():
                    full_param_name = f"bg.params.{param_name}"
                    optimizer = self.optimizers[full_param_name]
                    dup_in_optim(
                        optimizer,
                        [new_params],
                        should_bg_split,
                        num_bg_splits * 2 + num_bg_dups,
                    )

        # update running stats
        if not only_fg:
            for k, v in self.running_stats.items():
                v_fg, v_bg = v[:num_fg], v[num_fg:]
                new_v = torch.cat(
                    [
                        v_fg[~should_fg_split],
                        v_fg[should_fg_dup],
                        v_fg[should_fg_split].repeat(2),
                        v_bg[~should_bg_split],
                        v_bg[should_bg_dup],
                        v_bg[should_bg_split].repeat(2),
                    ],
                    dim=0,
                )
                self.running_stats[k] = new_v
        else:
            for k, v in self.running_stats.items():
                v_fg, v_bg = v[:num_fg], v[num_fg:]
                new_v = torch.cat(
                    [
                        v_fg[~should_fg_split],
                        v_fg[should_fg_dup],
                        v_fg[should_fg_split].repeat(2),
                        v_bg
                    ],
                    dim=0,
                )
                self.running_stats[k] = new_v


        guru.info(
            f"Split {should_split.sum().item()} gaussians, "
            f"Duplicated {should_dup.sum().item()} gaussians, "
            f"{self.model.num_gaussians} full gaussians left"
            f"{self.model.num_fg_gaussians} fg gaussians left"
            f"{self.model.num_bg_gaussians} bg gaussians left"
        )

    @torch.no_grad()
    def _cull_control_step(self, global_step, only_fg=False):
        # Cull gaussians.
        cfg = self.optim_cfg
        opacities = self.model.get_opacities_all()
        device = opacities.device
        is_opacity_too_small = opacities < cfg.cull_opacity_threshold
        is_radius_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        is_scale_too_big = torch.zeros_like(is_opacity_too_small, dtype=torch.bool)
        cull_scale_threshold = (
            torch.ones(len(is_scale_too_big), device=device) * cfg.cull_scale_threshold
        )
        
        num_fg = self.model.num_fg_gaussians
        cull_scale_threshold[num_fg:] *= self.model.bg_scene_scale
        if global_step > self.reset_opacity_every:
            scales = self.model.get_scales_all()
            is_scale_too_big = scales.amax(dim=-1) > cull_scale_threshold
            if global_step < cfg.stop_control_by_screen_steps:
                is_radius_too_big = (
                    self.running_stats["max_radii"] > cfg.cull_screen_threshold
                )
        should_cull = is_opacity_too_small | is_radius_too_big | is_scale_too_big
        should_fg_cull = should_cull[:num_fg]
        should_bg_cull = should_cull[num_fg:]

        fg_param_map = self.model.fg.cull_params(should_fg_cull)
        for param_name, new_params in fg_param_map.items():
            full_param_name = f"fg.params.{param_name}"
            optimizer = self.optimizers[full_param_name]
            remove_from_optim(optimizer, [new_params], should_fg_cull)

        if not only_fg:
            if self.model.bg is not None:
                bg_param_map = self.model.bg.cull_params(should_bg_cull)
                for param_name, new_params in bg_param_map.items():
                    full_param_name = f"bg.params.{param_name}"
                    optimizer = self.optimizers[full_param_name]
                    remove_from_optim(optimizer, [new_params], should_bg_cull)

        # update running stats
        if not only_fg:
            for k, v in self.running_stats.items():
                self.running_stats[k] = v[~should_cull]
        else:
            for k, v in self.running_stats.items():
                should_bg_cull = torch.zeros_like(should_bg_cull)
                should_cull = torch.cat([should_fg_cull, should_bg_cull], -1)
                self.running_stats[k] = v[~should_cull]


        guru.info(
            f"Culled {should_cull.sum().item()} gaussians, "
            f"{self.model.num_gaussians} full gaussians left"
            f"{self.model.num_fg_gaussians} fg gaussians left"
            f"{self.model.num_bg_gaussians} bg gaussians left"
        )

    @torch.no_grad()
    def _reset_opacity_control_step(self, only_fg=False):
        # Reset gaussian opacities.
        new_val = torch.logit(torch.tensor(0.8 * self.optim_cfg.cull_opacity_threshold))
        if not only_fg:
            for part in ["fg", "bg"]:
                part_params = getattr(self.model, part).reset_opacities(new_val)
                # Modify optimizer states by new assignment.
                for param_name, new_params in part_params.items():
                    full_param_name = f"{part}.params.{param_name}"
                    optimizer = self.optimizers[full_param_name]
                    reset_in_optim(optimizer, [new_params])
        else:
            for part in ["fg"]:
                part_params = getattr(self.model, part).reset_opacities(new_val)
                # Modify optimizer states by new assignment.
                for param_name, new_params in part_params.items():
                    full_param_name = f"{part}.params.{param_name}"
                    optimizer = self.optimizers[full_param_name]
                    reset_in_optim(optimizer, [new_params])
        guru.info("Reset opacities")

    def configure_optimizers(self):
        def _exponential_decay(step, *, lr_init, lr_final):
            t = np.clip(step / self.optim_cfg.max_steps, 0.0, 1.0)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init

        lr_dict = asdict(self.lr_cfg)
        optimizers = {}
        schedulers = {}
        # named parameters will be [part].params.[field]
        # e.g. fg.params.means
        # lr config is a nested dict for each fg/bg part
        for name, params in self.model.named_parameters():
            if name.split('.')[0] == "move_model":
                continue
            part, _, field = name.split(".")
            lr = lr_dict[part][field]
            optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])

            if "scales" in name:
                fnc = functools.partial(_exponential_decay, lr_final=0.1 * lr)
            else:
                fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )
        return optimizers, schedulers


def dup_in_optim(optimizer, new_params: list, should_dup: torch.Tensor, num_dups: int):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            p = param_state[key]
            param_state[key] = torch.cat(
                [p[~should_dup], p.new_zeros(num_dups, *p.shape[1:])],
                dim=0,
            )
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def remove_from_optim(optimizer, new_params: list, _should_cull: torch.Tensor):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            if key == "step":
                continue
            param_state[key] = param_state[key][~_should_cull]
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()


def reset_in_optim(optimizer, new_params: list):
    assert len(optimizer.param_groups) == len(new_params)
    for i, p_new in enumerate(new_params):
        old_params = optimizer.param_groups[i]["params"][0]
        param_state = optimizer.state[old_params]
        if len(param_state) == 0:
            return
        for key in param_state:
            param_state[key] = torch.zeros_like(param_state[key])
        del optimizer.state[old_params]
        optimizer.state[p_new] = param_state
        optimizer.param_groups[i]["params"] = [p_new]
        del old_params
        torch.cuda.empty_cache()
