import json
import os
import os.path as osp
from dataclasses import dataclass
from glob import glob
from itertools import product
from typing import Literal

import imageio.v3 as iio
import numpy as np
import roma
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import flow3d.data.colmap as colmap

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.colmap import get_colmap_camera_params
from flow3d.data.utils import (
    SceneNormDict,
    masked_median_blur,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4


@dataclass
class StereoHighDataConfig:
    data_dir: str
    start: int = 0  # 0
    end: int = 24   # -1
    factor: int = 1
    split: Literal["train", "val"] = "train"
    depth_type: Literal[
        "midas",
        "depth_anything",
        "lidar",
        "depth_anything_colmap",
    ] = "depth_anything_colmap" 
    camera_type: Literal["refined"] = "refined"
    use_median_filter: bool = False
    num_targets_per_frame: int = 4
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    load_from_cache: bool = True
    skip_load_imgs: bool = False
    image_dir = "images"


class StereoHighDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        start: int = 0,
        end: int = 24,
        factor: int = 1,
        split: Literal["train", "val"] = "train",
        depth_type: Literal[
            "midas",
            "depth_anything",
            "lidar",
            "depth_anything_colmap",
        ] = "depth_anything_colmap", 
        camera_type: Literal["original", "refined"] = "refined",
        use_median_filter: bool = False,
        num_targets_per_frame: int = 1,
        scene_norm_dict: SceneNormDict | None = None,
        load_from_cache: bool = False,
        skip_load_imgs: bool = False,
        image_dir = "images",
        **_,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.training = split == "train"
        self.split = split
        self.factor = factor
        self.start = start
        self.end = end
        self.depth_type = depth_type
        self.camera_type = camera_type
        self.use_median_filter = use_median_filter
        self.num_targets_per_frame = num_targets_per_frame
        self.scene_norm_dict = scene_norm_dict
        self.load_from_cache = load_from_cache
        self.cache_dir = osp.join(data_dir, "flow3d_preprocessed", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.has_validation = True
        self.image_dir = image_dir

        # Load metadata.    
        image_all_paths = glob.glob(os.path.join(data_dir, self.image_dir, "*.png"))
        image_all_paths = sorted(image_all_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))


        if split == "train":
            image_paths = image_all_paths[::2] 
            self.frame_names = [f.split('/')[-1].split('.')[0] for f in image_paths]
            time_ids = [ii for ii in range(0, len(image_paths))]
            self.time_ids = torch.tensor(time_ids)
            full_len = len(image_paths)
            end = min(end, full_len) if end > 0 else full_len
            self.start = start
            self.end = end
            self.frame_names = self.frame_names[0:24]
            self.time_ids = self.time_ids[0:24]
        else:
            image_paths = image_all_paths
            times = [ii//2 for ii in range(0, len(image_paths))]

            full_len = len(image_paths)
            end = min(end, full_len) if end > 0 else full_len
            self.end = end*2
            self.start = self.start*2

            self.frame_names = [f.split('/')[-1].split('.')[0] for f in image_paths[self.start:self.end]]
            time_ids = [t for t in times[self.start:self.end]]
            self.time_ids = torch.tensor(time_ids)


        guru.info(f"{self.time_ids.min()=} {self.time_ids.max()=}")
        guru.info(f"{self.num_frames=}")
        self.fps = 10.  

        # Load cameras.
        if self.camera_type == "refined":
            Ks, w2cs = get_colmap_camera_params(
                osp.join(data_dir, "flow3d_preprocessed/colmap/sparse/"),
                [frame_name + ".png" for frame_name in self.frame_names],
            )
            self.Ks = torch.from_numpy(Ks[:, :3, :3].astype(np.float32))
            self.Ks[:, :2] /= factor
            self.w2cs = torch.from_numpy(w2cs.astype(np.float32))

            if self.w2cs.shape[0] > 24 and split == "train":
                self.w2cs = self.w2cs[0:24]
                self.Ks = self.Ks[0:24]
            if self.w2cs.shape[0] > 24 and split != "train":
                self.w2cs = self.w2cs[0:48]
                self.Ks = self.Ks[0:48]

        if not skip_load_imgs:
            # Load images.
            imgs = torch.from_numpy(
                np.array(
                    [
                        iio.imread(
                            osp.join(self.data_dir, f"{self.image_dir}/{frame_name}.png")
                        )
                        for frame_name in tqdm(
                            self.frame_names,
                            desc=f"Loading {self.split} images",
                            leave=False,
                        )
                    ],
                )
            )
            self.imgs = imgs[..., :3] / 255.0

            self.valid_masks =  np.ones_like(imgs[..., 0]) 
            # Load masks.
            self.masks = (
                torch.from_numpy(
                    np.array(
                        [
                            iio.imread(
                                osp.join(
                                    self.data_dir,
                                    "flow3d_preprocessed/masks/",
                                    f"{frame_name}.png",
                                )                          
                            )
                            for frame_name in tqdm(
                                self.frame_names,
                                desc=f"Loading {self.split} masks",
                                leave=False,
                            )
                        ],
                    )
                )
                / 255.0
            )
            if self.masks.shape[-1] == 3:
                self.masks = self.masks[:, :, :, 0]
            # Load depths.
            def load_depth(frame_name):
                depth = np.load(
                    osp.join(
                        self.data_dir,
                        f"flow3d_preprocessed/aligned_{self.depth_type}/",
                        f"{frame_name}.npy",
                    )
                )
                depth[depth < 1e-3] = 1e-3
                depth = 1.0 / depth
                return depth

            self.depths = torch.from_numpy(
                np.array(
                    [
                        load_depth(frame_name)
                        for frame_name in tqdm(
                            self.frame_names,
                            desc=f"Loading {self.split} depths",
                            leave=False,
                        )
                    ],
                    np.float32,
                )
            )
            max_depth_values_per_frame = self.depths.reshape(
                self.num_frames, -1
            ).max(1)[0]
            max_depth_value = max_depth_values_per_frame.median() * 2.5
            self.depths = torch.clamp(self.depths, 0, max_depth_value)
            # Median filter depths.
            # NOTE(hangg): This operator is very expensive.
            if self.use_median_filter:
                for i in tqdm(
                    range(self.num_frames), desc="Processing depths", leave=False
                ):
                    depth = masked_median_blur(
                        self.depths[[i]].unsqueeze(1).to("cuda"),
                        (
                            self.masks[[i]]
                            * self.valid_masks[[i]]
                            * (self.depths[[i]] > 0)
                        )
                        .unsqueeze(1)
                        .to("cuda"),
                    )[0, 0].cpu()
                    self.depths[i] = depth * self.masks[i] + self.depths[i] * (
                        1 - self.masks[i]
                    )

            if self.training:
                # Load the query pixels from 2D tracks.
                self.query_tracks_2d = [
                    torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "flow3d_preprocessed/2d_tracks/",
                                f"{frame_name}_{frame_name}.npy",
                            )
                        ).astype(np.float32)
                    )
                    for frame_name in self.frame_names
                ]
                guru.info(
                    f"{len(self.query_tracks_2d)=} {self.query_tracks_2d[0].shape=}"
                )

        # None
        if self.scene_norm_dict is None:
            cached_scene_norm_dict_path = osp.join(
                self.cache_dir, "scene_norm_dict.pth"
            )
            if osp.exists(cached_scene_norm_dict_path) and self.load_from_cache:
                print("loading cached scene norm dict...")
                self.scene_norm_dict = torch.load(
                    osp.join(self.cache_dir, "scene_norm_dict.pth")
                )
            elif self.training:
                # Compute the scene scale and transform for normalization.
                # Normalize the scene based on the foreground 3D tracks.
                num_dyn_frames = len(self.get_dyn_time_ids())
                subsampled_tracks_3d = self.get_tracks_3d(
                    num_samples=10000, step=num_dyn_frames//4, show_pbar=False
                )[0]
                scene_center = subsampled_tracks_3d.mean((0, 1))
                tracks_3d_centered = subsampled_tracks_3d - scene_center
                min_scale = tracks_3d_centered.quantile(0.05, dim=0)
                max_scale = tracks_3d_centered.quantile(0.95, dim=0)
                scale = torch.max(max_scale - min_scale).item() / 2.0
                original_up = -F.normalize(self.w2cs[:, 1, :3].mean(0), dim=-1)
                target_up = original_up.new_tensor([0.0, 0.0, 1.0])
                R = roma.rotvec_to_rotmat(
                    F.normalize(original_up.cross(target_up, dim=-1), dim=-1)
                    * original_up.dot(target_up).acos_()
                )
                transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
                self.scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)
                torch.save(self.scene_norm_dict, cached_scene_norm_dict_path)
            else:
                raise ValueError("scene_norm_dict must be provided for validation.")

        # Normalize the scene.
        scale = self.scene_norm_dict["scale"]
        transfm = self.scene_norm_dict["transfm"]
        self.w2cs = self.w2cs @ torch.linalg.inv(transfm)
        self.w2cs[:, :3, 3] /= scale
        if self.training and not skip_load_imgs:
            self.depths /= scale

        if not skip_load_imgs:
            guru.info(
                f"{self.imgs.shape=} {self.valid_masks.shape=} {self.masks.shape=}"
            )


    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    def __len__(self):
        return self.imgs.shape[0]
    
    def get_dyn_time_ids(self):            
        return self.time_ids[self.start:self.end] - self.start  # 0, 1, 2,...
    
    def get_dyn_image_ids(self):   
        images_ids = [id for id in range(0, len(self.frame_names))]
        return images_ids[self.start:self.end]

    def get_w2cs(self) -> torch.Tensor:
        return self.w2cs

    def get_Ks(self) -> torch.Tensor:
        return self.Ks
    
    def get_w2cs_dyn(self) -> torch.Tensor:
        return self.w2cs[self.start:self.end]

    def get_Ks_dyn(self) -> torch.Tensor:
        return self.Ks[self.start:self.end]

    def get_image(self, index: int) -> torch.Tensor:
        return self.imgs[index]

    def get_depth(self, index: int) -> torch.Tensor:
        return self.depths[index]

    def get_masks(self, index: int) -> torch.Tensor:
        return self.masks[index]

    def get_img_wh(self):
        return iio.imread(
            osp.join(self.data_dir, f"{self.image_dir}/{self.frame_names[0]}.png")
        ).shape[1::-1]

    # def get_sam_features(self) -> list[torch.Tensor, tuple[int, int], tuple[int, int]]:
    #     return self.sam_features, self.sam_original_size, self.sam_input_size

    def get_tracks_3d(
        self, num_samples: int, step: int = 1, show_pbar: bool = True, **kwargs
    ):
        """Get 3D tracks from the dataset.

        Args:
            num_samples (int | None): The number of samples to fetch. If None,
                fetch all samples. If not None, fetch roughly a same number of
                samples across each frame. Note that this might result in
                number of samples less than what is specified.
            step (int): The step to temporally subsample the track.
        """
        assert (
            self.split == "train"
        ), "fetch_tracks_3d is only available for the training split."
        cached_track_3d_path = osp.join(self.cache_dir, f"tracks_3d_{num_samples}.pth")
        # Load 2D tracks.
        num_dyn_frames = len(self.get_dyn_time_ids())
        raw_tracks_2d = []
        candidate_frames = list(range(self.start, self.end, step))
        print(candidate_frames, self.start, self.end)
        num_sampled_frames = len(candidate_frames)
        for i in (
            tqdm(candidate_frames, desc="Loading 2D tracks", leave=False)
            if show_pbar
            else candidate_frames
        ):
            curr_num_samples = self.query_tracks_2d[i].shape[0]   
            num_samples_per_frame = (
                int(np.floor(num_samples / num_sampled_frames))
                if i != candidate_frames[-1]
                else num_samples
                - (num_sampled_frames - 1)
                * int(np.floor(num_samples / num_sampled_frames))
            )
            if num_samples_per_frame < curr_num_samples:
                track_sels = np.random.choice(
                    curr_num_samples, (num_samples_per_frame,), replace=False
                )
            else:
                track_sels = np.arange(0, curr_num_samples)

            curr_tracks_2d = []
            for j in range(self.start, self.end, step):
                if i == j:
                    target_tracks_2d = self.query_tracks_2d[i]
                else:
                    target_tracks_2d = torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "flow3d_preprocessed/2d_tracks/",
                                f"{self.frame_names[i]}_"
                                f"{self.frame_names[j]}.npy",
                            )
                        ).astype(np.float32)
                    )
                curr_tracks_2d.append(target_tracks_2d[track_sels])
            raw_tracks_2d.append(torch.stack(curr_tracks_2d, dim=1))
        for ii in range(0, num_sampled_frames):
            guru.info(f"{step=} {len(raw_tracks_2d)=} {raw_tracks_2d[ii].shape=}")

        inv_Ks = torch.linalg.inv(self.Ks[self.start:self.end])[::step]
        c2ws = torch.linalg.inv(self.w2cs[self.start:self.end])[::step]
        H, W = self.imgs.shape[1:3]
        filtered_tracks_3d, filtered_visibles, filtered_track_colors = [], [], []
        filtered_invisibles, filtered_confidences = [], []
        masks = self.masks[self.start:self.end] * self.valid_masks[self.start:self.end] * (self.depths[self.start:self.end] > 0)
        masks = (masks > 0.5).float()
        for i, tracks_2d in enumerate(raw_tracks_2d):
            tracks_2d = tracks_2d.swapdims(0, 1)
            tracks_2d, occs, dists = (
                tracks_2d[..., :2],
                tracks_2d[..., 2],
                tracks_2d[..., 3],
            )
            # visibles = postprocess_occlusions(occs, dists)
            visibles, invisibles, confidences = parse_tapir_track_info(occs, dists)
            # Unproject 2D tracks to 3D.
            track_depths = F.grid_sample(
                self.depths[self.start:self.end][::step, None],
                normalize_coords(tracks_2d[..., None, :], H, W),
                align_corners=True,
                padding_mode="border",
            )[:, 0]

            tracks_3d = (
                torch.einsum(
                    "nij,npj->npi",
                    inv_Ks,
                    F.pad(tracks_2d, (0, 1), value=1.0),
                )
                * track_depths
            )
            tracks_3d = torch.einsum(
                "nij,npj->npi", c2ws, F.pad(tracks_3d, (0, 1), value=1.0)
            )[..., :3]
            # Filter out out-of-mask tracks.
            is_in_masks = (
                F.grid_sample(
                    masks[::step, None],
                    normalize_coords(tracks_2d[..., None, :], H, W),
                    align_corners=True,
                ).squeeze()
                == 1
            )
            visibles *= is_in_masks
            invisibles *= is_in_masks
            confidences *= is_in_masks.float()
            # Get track's color from the query frame.
            track_colors = (
                F.grid_sample(
                    self.imgs[self.start:self.end][i * step : i * step + 1].permute(0, 3, 1, 2),
                    normalize_coords(tracks_2d[i : i + 1, None, :], H, W),
                    align_corners=True,
                    padding_mode="border",
                )
                .squeeze()
                .T
            )
            # at least visible 5% of the time, otherwise discard
            visible_counts = visibles.sum(0)
            valid = visible_counts >= min(
                int(0.05 * num_dyn_frames),
                visible_counts.float().quantile(0.1).item(),
            )

            filtered_tracks_3d.append(tracks_3d[:, valid])
            filtered_visibles.append(visibles[:, valid])
            filtered_invisibles.append(invisibles[:, valid])
            filtered_confidences.append(confidences[:, valid])
            filtered_track_colors.append(track_colors[valid])

        filtered_tracks_3d = torch.cat(filtered_tracks_3d, dim=1).swapdims(0, 1)
        filtered_visibles = torch.cat(filtered_visibles, dim=1).swapdims(0, 1)
        filtered_invisibles = torch.cat(filtered_invisibles, dim=1).swapdims(0, 1)
        filtered_confidences = torch.cat(filtered_confidences, dim=1).swapdims(0, 1)
        filtered_track_colors = torch.cat(filtered_track_colors, dim=0)

        guru.info(f"tracking data: {filtered_tracks_3d.shape} {filtered_visibles.shape} {filtered_invisibles.shape} {filtered_confidences.shape} {filtered_track_colors.shape}")
        if step == 1:
            torch.save(
                {
                    "tracks_3d": filtered_tracks_3d,
                    "visibles": filtered_visibles,
                    "invisibles": filtered_invisibles,
                    "confidences": filtered_confidences,
                    "track_colors": filtered_track_colors,
                },
                cached_track_3d_path,
            )
        return (
            filtered_tracks_3d,
            filtered_visibles,
            filtered_invisibles,
            filtered_confidences,
            filtered_track_colors,
        )

    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) :
        H, W = self.imgs.shape[1:3]
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(W, dtype=torch.float32),
                torch.arange(H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )
        candidate_frames = list(range(self.num_frames))
        num_sampled_frames = len(candidate_frames)
        bkgd_points, bkgd_point_normals, bkgd_point_colors = [], [], []
        for i in tqdm(candidate_frames, desc="Loading bkgd points", leave=False):
            img = self.imgs[i]
            depth = self.depths[i]
            bool_mask = ((1.0 - self.masks[i]) * self.valid_masks[i] * (depth > 0)).to(
                torch.bool
            )
            w2c = self.w2cs[i]
            K = self.Ks[i]
            points = (
                torch.einsum(
                    "ij,pj->pi",
                    torch.linalg.inv(K),
                    F.pad(grid[bool_mask], (0, 1), value=1.0),
                )
                * depth[bool_mask][:, None]
            )
            points = torch.einsum(
                "ij,pj->pi", torch.linalg.inv(w2c)[:3], F.pad(points, (0, 1), value=1.0)
            )
            point_normals = normal_from_depth_image(depth, K, w2c)[bool_mask]
            point_colors = img[bool_mask]
            curr_num_samples = points.shape[0]
            num_samples_per_frame = (
                int(np.floor(num_samples / num_sampled_frames))
                if i != candidate_frames[-1]
                else num_samples
                - (num_sampled_frames - 1)
                * int(np.floor(num_samples / num_sampled_frames))
            )
            if num_samples_per_frame < curr_num_samples:
                point_sels = np.random.choice(
                    curr_num_samples, (num_samples_per_frame,), replace=False
                )
            else:
                point_sels = np.arange(0, curr_num_samples)
            bkgd_points.append(points[point_sels])
            bkgd_point_normals.append(point_normals[point_sels])
            bkgd_point_colors.append(point_colors[point_sels])
        bkgd_points = torch.cat(bkgd_points, dim=0)
        bkgd_point_normals = torch.cat(bkgd_point_normals, dim=0)
        bkgd_point_colors = torch.cat(bkgd_point_colors, dim=0)

        return bkgd_points, bkgd_point_normals, bkgd_point_colors

    def get_video_dataset(self) -> Dataset:
        return StereoHighDatasetVideoView(self)

    def __getitem__(self, index: int):
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": self.time_ids[index],
            # (4, 4).
            "w2cs": self.w2cs[index],
            # (3, 3).
            "Ks": self.Ks[index],
            # (H, W, 3).
            "imgs": self.imgs[index],
            # (H, W).
            "valid_masks": self.valid_masks[index],
            # (H, W).
            "masks": self.masks[index],

        }

        if self.training:
            data["start"] = self.start
        else:
            data["start"] = self.start//2
        
        # (H, W).
        data["depths"] = self.depths[index]

        if self.training:
            # (P, 2).
            data["query_tracks_2d"] = self.query_tracks_2d[index][:, :2]
            target_inds = torch.from_numpy(
                np.random.choice(
                    self.get_dyn_image_ids(), (self.num_targets_per_frame,), replace=False
                )
            )
            # (N, P, 4).
            target_tracks_2d = torch.stack(
                [
                    torch.from_numpy(
                        np.load(
                            osp.join(
                                self.data_dir,
                                "flow3d_preprocessed/2d_tracks/",
                                f"{self.frame_names[index]}_"
                                f"{self.frame_names[target_index.item()]}.npy",
                            )
                        ).astype(np.float32)
                    )
                    for target_index in target_inds
                ],
                dim=0,
            )
            # (N,).
            target_ts = self.time_ids[target_inds]

            data["target_ts"] = target_ts
            # (N, 4, 4).
            data["target_w2cs"] = self.w2cs[target_ts]
            # (N, 3, 3).
            data["target_Ks"] = self.Ks[target_ts]
            # (N, P, 2).
            data["target_tracks_2d"] = target_tracks_2d[..., :2]
            # (N, P).
            (
                data["target_visibles"],
                data["target_invisibles"],
                data["target_confidences"],
            ) = parse_tapir_track_info(
                target_tracks_2d[..., 2], target_tracks_2d[..., 3]
            )
            # (N, P).
            data["target_track_depths"] = F.grid_sample(
                self.depths[target_inds, None],
                normalize_coords(
                    target_tracks_2d[..., None, :2],
                    self.imgs.shape[1],
                    self.imgs.shape[2],
                ),
                align_corners=True,
                padding_mode="border",
            )[:, 0, :, 0]

            data["target_track_imgs"] = F.grid_sample(
                self.imgs[target_inds].permute(0, 3, 1, 2),
                normalize_coords(
                    target_tracks_2d[..., None, :2],
                    self.imgs.shape[1],
                    self.imgs.shape[2],
                ),
                align_corners=True,
                padding_mode="border",
            ).squeeze(-1)
        return data

    def preprocess(self, data):
        return data


class StereoHighDatasetVideoView(Dataset):
    """Return a dataset view of the video trajectory."""

    def __init__(self, dataset: StereoHighDataset):
        super().__init__()
        self.dataset = dataset
        self.fps = self.dataset.fps
        assert self.dataset.split == "train"

    def __len__(self):
        return self.dataset.num_frames

    def __getitem__(self, index):
        return {
            "frame_names": self.dataset.frame_names[index],
            "ts": index,
            "w2cs": self.dataset.w2cs[index],
            "Ks": self.dataset.Ks[index],
            "imgs": self.dataset.imgs[index],
            "depths": self.dataset.depths[index],
            "masks": self.dataset.masks[index],
        }

