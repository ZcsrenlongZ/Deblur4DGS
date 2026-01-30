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
from flow3d.params import GaussianParams, MotionBases
import glob
import imageio.v3 as iio
import cv2
import torch.nn.functional as F


torch.set_float32_matmul_precision("high")
def set_seed(seed):
    # Set the seed for generating random numbers
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)


def adaptive_slides(data_dir):
    image_files = glob.glob(os.path.join(data_dir, 'images', '*.png'))
    image_files = sorted(image_files, key=lambda x:int(x.split('/')[-1].split('.')[0]))[::2]
    mask_files = glob.glob(os.path.join(data_dir, 'masks', '*.png'))
    mask_files = sorted(mask_files, key=lambda x:int(x.split('/')[-1].split('.')[0]))[::2]
    assert len(image_files) == len(mask_files)
    scores = []
    for ii in range(0, len(image_files)):
        image = cv2.imread(image_files[ii])
        image = np.mean(image, -1)
        mask = cv2.imread(mask_files[ii])/255.
        mask = mask[:, :, 0]
        image = image * mask
        image_lp = cv2.Laplacian(image, cv2.CV_64F)
        inter_image = image_lp - (np.sum(image_lp)/np.sum(mask))
        score = np.sum(inter_image * inter_image)/np.sum(mask)
        scores.append(score)

    
    scores = np.array(scores)
    slides = {'0':[ 0, 1, 2, 3, 4],  
              '1':[ 5, 6, 7, 8, 9],
              '2':[10,11,12,13,14],
              '3':[15,16,17,18,19],
              '4':[19,20,21,22,23]}
    trys = {'0':[0, 4], 
            '1':[2, 2],
            '2':[2, 2],
            '3':[2, 2],
            '4':[4, 0]}
    for ii in range(0, len(slides.keys())):
        id = list(slides.keys())[ii]
        left_try_scores = scores[slides[id][0]-trys[id][0]:slides[id][0]+1]
        right_try_scores = scores[slides[id][-1]:slides[id][-1]+trys[id][-1]+1]
        extend_id = slides[id][0] - (trys[id][0] - np.argmax(left_try_scores)) - 1
        for jj in range(slides[id][0]-1, extend_id, -1):
            slides[id].insert(0, jj)

        extend_id = np.argmax(right_try_scores)+slides[id][-1]
        for jj in range(slides[id][-1]+1, extend_id+1):
            slides[id].append(jj)

    score_dict = {}
    for ii in range(0, len(slides.keys())):
        id = list(slides.keys())[ii]
        score_dict[id] = scores[slides[id][0]:slides[id][-1]+1] 
        assert len(score_dict[id]) == len(slides[id])
    slides_dict = slides
    return slides_dict, score_dict


@dataclass
class TrainConfig:
    work_dir: str
    data: (
          Annotated[StereoLowDataConfig, tyro.conf.subcommand(name="stereolow")]
        | Annotated[StereoHighDataConfig, tyro.conf.subcommand(name="stereohigh")]
    )
    lr: SceneLRConfig
    loss: LossesConfig
    optim: OptimizerConfig
    num_fg: int = 40_000                            
    num_bg: int = 100_000                          
    num_motion_bases: int = 20  
    num_epochs: int = 200     
    port: int | None = None
    vis_debug: bool = False 
    batch_size: int = 1 
    num_dl_workers: int = 4
    validate_every: int = 50
    save_videos_every: int = 50


def main(cfg: TrainConfig):

    work_dir = cfg.work_dir
    data_dir = cfg.data.data_dir

    # optimization current scale
    cfg.work_dir = work_dir
    cfg.data.data_dir= data_dir
    scale = 'x1'

    slides_dict, score_dict = adaptive_slides(os.path.join(cfg.data.data_dir, 'x1'))
    assert len(slides_dict) == len(score_dict)
    for jj in range(0, len(slides_dict.keys())):

        name = list(slides_dict.keys())[jj]
        cfg.data.start = slides_dict[name][0]
        cfg.data.end = slides_dict[name][-1] + 1
        cfg.data.data_dir = os.path.join(data_dir, scale)
        cfg.data.factor = int(scale.split('x')[-1])
        cfg.work_dir = os.path.join(work_dir, scale, '%02d'%cfg.data.start+'%02d'%cfg.data.end)
        backup_code(cfg.work_dir)     
        train_dataset, train_video_view, val_img_dataset, val_kpt_dataset = (
            get_train_val_datasets(cfg.data, load_val=True)
        )
        guru.info(f"Training dataset has {train_dataset.num_frames} frames")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
        ckpt_path = f"{cfg.work_dir}/ckpts/100.ckpt"
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

        trainer.model.training = False
        val_logs = validator.validate_imgs_with_optimization()
        # val_logs = validator.validate_imgs(epoch=199)



def initialize_and_checkpoint_model_from_static(
    cfg: TrainConfig,
    train_dataset: BaseDataset,
    device: torch.device,
    ckpt_path: str,
    vis: bool = False,
    port: int | None = None,
    scores=None
):
    
    fg_params, motion_bases, bg_params, tracks_3d = init_model_from_tracks(
        train_dataset,
        cfg.num_fg,
        cfg.num_bg,
        cfg.num_motion_bases,
        vis=vis,
        port=port,
        scores=scores
    )

    ckpt = torch.load(ckpt_path)
    mode_dict = ckpt["model"]
    bg_params = GaussianParams(
    mode_dict['bg.params.means'],
    mode_dict['bg.params.quats'],
    mode_dict['bg.params.scales'],
    mode_dict['bg.params.colors'],
    mode_dict['bg.params.opacities'],
    scene_center=mode_dict['bg.scene_center'],
    scene_scale=mode_dict['bg.scene_scale'])

    move_model_dict = ckpt['move_model']
    Ks = train_dataset.get_Ks_dyn().to(device)      
    w2cs = train_dataset.get_w2cs_dyn().to(device)   

    run_initial_optim(fg_params, motion_bases, tracks_3d, Ks, w2cs)
    if vis and cfg.port is not None:
        server = get_server(port=cfg.port)
        vis_init_params(server, fg_params, motion_bases)

    model = SceneModel(Ks, w2cs, fg_params, motion_bases, bg_params)   

    ckpt_path = f"{cfg.work_dir}/checkpoints/last.ckpt"
    guru.info(f"Saving initialization to {ckpt_path}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model": model.state_dict(), 
                "epoch": 0, 
                "global_step": 0,
                "move_model":move_model_dict
                }, 
                ckpt_path)


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
    scores=None
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
    # 
    if scores is not None:
        cano_t = np.argmax(scores)
        guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=} score:{scores[cano_t]} scores:{scores}")
    else:
        cano_t = int(tracks_3d.visibles.sum(dim=0).argmax().item()) 
        guru.info(f"{cano_t=} {num_fg=} {num_bg=} {num_motion_bases=} vis:{tracks_3d.visibles.sum(dim=0)[cano_t]} viss:{tracks_3d.visibles.sum(dim=0)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    motion_bases, motion_coefs, tracks_3d = init_motion_params_with_procrustes(
        tracks_3d, num_motion_bases, rot_type, cano_t, vis=vis, port=port
    )
    motion_bases = motion_bases.to(device)

    fg_params = init_fg_from_tracks_3d(cano_t, tracks_3d, motion_coefs)
    fg_params = fg_params.to(device)


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
