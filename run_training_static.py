import os
import os.path as osp
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated

import numpy as np
import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
from flow3d.data import (
    BaseDataset,
    get_train_val_datasets,
    StereoLowDataConfig,
    StereoHighDataConfig,


)
from flow3d.data.utils import to_device
from flow3d.init_utils import (
    init_bg,
    init_fg_from_tracks_3d,
    init_motion_params_with_procrustes,
    run_initial_optim,
    vis_init_params,
)
from flow3d.scene_model import SceneModel
from flow3d.tensor_dataclass import StaticObservations, TrackObservations
from flow3d.trainer import Trainer
from flow3d.validator import Validator
from flow3d.vis.utils import get_server
import random

torch.set_float32_matmul_precision("high")


def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


@dataclass
class TrainConfig:
    work_dir: str
    data: (Annotated[StereoLowDataConfig, tyro.conf.subcommand(name="stereolow")]
        | Annotated[StereoHighDataConfig, tyro.conf.subcommand(name="stereohigh")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 50      # 100                      
    num_bg: int = 100_000                           
    num_motion_bases: int = 20  
    num_epochs: int = 400     
    port: int | None = None
    vis_debug: bool = False  
    batch_size: int = 1        
    num_dl_workers: int = 4
    validate_every: int = 100
    save_videos_every: int = 100


def main(cfg: TrainConfig):
    work_dir = cfg.work_dir
    data_dir = cfg.data.data_dir
    scales = ['x1']
    for ss in range(0, len(scales)):
        scale = scales[ss]
        cfg.data.data_dir = os.path.join(data_dir, scale)
        cfg.data.factor = int(scale.split('x')[-1])
        cfg.work_dir = os.path.join(work_dir, scale, '0023')
        backup_code(cfg.work_dir)     

        train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
            get_train_val_datasets(cfg.data, load_val=True)
        )
        guru.info(f"Training dataset has {train_dataset.num_frames} frames")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # save config
        os.makedirs(cfg.work_dir, exist_ok=True)
        with open(f"{cfg.work_dir}/cfg.yaml", "w") as f:
            yaml.dump(asdict(cfg), f, default_flow_style=False)

        ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
        initialize_and_checkpoint_model(
            cfg,
            train_dataset,
            device,
            ckpt_path,
            vis=cfg.vis_debug,
            port=cfg.port,
        )

        trainer, start_epoch = Trainer.init_from_checkpoint(
            ckpt_path,
            device,
            cfg.lr,
            cfg.loss,
            cfg.optim,
            work_dir=cfg.work_dir,
            port=cfg.port,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_dl_workers,
            persistent_workers=True,
            collate_fn=BaseDataset.train_collate_fn,
        )

        dyn_time_ids = train_dataset.get_dyn_time_ids()
        validator = None
        if (
            train_video_view is not None
            or val_img_dataset is not None
            or val_kpt_dataset is not None
        ):
            validator = Validator(
                model=trainer.model,
                device=device,
                train_loader=(
                    DataLoader(train_video_view, batch_size=1) if train_video_view else None
                ),
                val_img_loader=(
                    DataLoader(val_img_dataset, batch_size=1) if val_img_dataset else None
                ),
                val_kpt_loader=(
                    DataLoader(val_kpt_dataset, batch_size=1) if val_kpt_dataset else None
                ),
                save_dir=cfg.work_dir,
            )

        guru.info(f"Starting training from {trainer.global_step=}")
        batches_sta = []
        for id in range(0, len(train_dataset.frame_names)):
            batch1 = to_device(train_dataset.__getitem__(id), device)
            batch1['frame_names'] = [str(batch1['frame_names'])]
            batch1['ts'] = batch1['ts'].unsqueeze(0)
            batch1['w2cs'] = batch1['w2cs'].unsqueeze(0)
            batch1['Ks'] = batch1['Ks'].unsqueeze(0)
            batch1['imgs'] = batch1['imgs'].unsqueeze(0)
            batch1['masks'] = batch1['masks'].unsqueeze(0)
            batch1['valid_masks'] = torch.FloatTensor(batch1['valid_masks']).unsqueeze(0).to(batch1['imgs'].device)
            batch1['depths'] = batch1['depths'].unsqueeze(0)
            batch1['query_tracks_2d'] = [batch1['query_tracks_2d']]
            batch1['target_ts'] = [batch1['target_ts']- train_dataset.start]
            batch1['target_w2cs'] = [batch1['target_w2cs']]
            batch1['target_Ks'] = [batch1['target_Ks']]
            batch1['target_tracks_2d'] = [batch1['target_tracks_2d']]
            batch1['target_visibles'] = [batch1['target_visibles']]
            batch1['target_invisibles'] = [batch1['target_invisibles']]
            batch1['target_confidences'] = [batch1['target_confidences']]
            batch1['target_track_depths'] = [batch1['target_track_depths']]
            batch1['target_track_imgs'] = [batch1['target_track_imgs']]
            batches_sta.append(batch1)

        for epoch in (
            pbar := tqdm(
                range(start_epoch, cfg.num_epochs),
                initial=start_epoch,
                total=cfg.num_epochs,
            )
        ):
            trainer.set_epoch(epoch)
            for ii in range(len(train_loader)):
                index = random.randint(1, len(batches_sta)-2)
                batch1 =[batches_sta[index-1], batches_sta[index], batches_sta[index+1]]
                loss = trainer.train_step(batch1=batch1, batch2=None, batch3=None, epoch=epoch, dyn_time_ids=dyn_time_ids, stage="first")
                pbar.set_description(f"Loss: {loss:.6f}")

            if validator is not None:
                if (epoch > 0 and epoch % cfg.validate_every == 0) or (
                    epoch == cfg.num_epochs - 1
                ):
                    trainer.model.training = False
                    val_logs, _ = validator.validate(epoch, mode='mid')  # 存deblur图
                    trainer.model.training = True
                    trainer.log_dict(val_logs)

            if (epoch+1) % cfg.validate_every ==0 :
                os.makedirs(f"{trainer.work_dir}/ckpts/static", exist_ok=True)
                trainer.save_checkpoint(f"{trainer.work_dir}/ckpts/static/{epoch}.ckpt")


def initialize_and_checkpoint_model(
    cfg: TrainConfig,
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
):
    if os.path.exists(ckpt_path):
        guru.info(f"model checkpoint exists at {ckpt_path}")
        return

    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
        train_dataset,
        cfg.num_fg,
        cfg.num_bg,
        cfg.num_motion_bases,
        vis=vis,
        port=port,
    )
    # run initial optimization
    Ks = train_dataset.get_Ks_dyn().to(device)     
    w2cs = train_dataset.get_w2cs_dyn().to(device)

    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs, num_iters=1000)
    if vis and cfg.port is not None:
        server = get_server(port=cfg.port)
        vis_init_params(server, fg_params, motion_bases)
    model = SceneModel(Ks, w2cs, fg_params, motion_bases, bg_params)  
    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": 0, "global_step": 0}, ckpt_path)


def  init_model_from_tracks(
    train_dataset,
    num_fg: int,
    num_bg: int,
    num_motion_bases: int,
    vis: bool = False,
    port: int | None = None,
):
    tracks_3d = TrackObservations(*train_dataset.get_tracks_3d(num_fg))
    print(
        f"{tracks_3d.xyz.shape=} {tracks_3d.visibles.shape=} "
        f"{tracks_3d.invisibles.shape=} {tracks_3d.confidences.shape} "
        f"{tracks_3d.colors.shape}"
    )
    if not tracks_3d.check_sizes():
        import ipdb

        ipdb.set_trace()
    rot_type = "6d"
    cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item())  
    guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)

    fg_params.params["opacities"] = fg_params.params["opacities"] * 0.

    bg_params = None
    if num_bg > 0:
        bg_points = StaticObservations(*train_dataset.get_bkgd_points(num_bg))
        assert bg_points.check_sizes()
        bg_params = init_bg(bg_points)
        bg_params = bg_params.to(device)

    tracks_3d = tracks_3d.to(device)

    return fg_params, motion_bases, bg_params, tracks_3d


def backup_code(work_dir):
    root_dir = osp.abspath(osp.join(osp.dirname(__file__)))
    tracked_dirs = [osp.join(root_dir, dirname) for dirname in ["flow3d", "scripts"]]
    dst_dir = osp.join(work_dir, "code", datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    for tracked_dir in tracked_dirs:
        if osp.exists(tracked_dir):
            shutil.copytree(tracked_dir, osp.join(dst_dir, osp.basename(tracked_dir)))


if __name__ == "__main__":
    main(tyro.cli(TrainConfig))
