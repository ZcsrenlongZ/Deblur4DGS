from dataclasses import asdict, replace

from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .stereo_low_dataset import (
    StereoLowDataConfig,
    StereoLowDataset,
    StereoLowDatasetVideoView
)
from .stereo_high_dataset import (
    StereoHighDataConfig,
    StereoHighDataset,
    StereoHighDatasetVideoView
)

def get_train_val_datasets(
    data_cfg: \
    StereoLowDataConfig | StereoHighDataConfig, 
    load_val: bool
) -> tuple[BaseDataset, Dataset | None, Dataset | None, Dataset | None]:
    train_video_view = None
    val_img_dataset = None
    val_kpt_dataset = None
    if (isinstance(data_cfg, StereoLowDataConfig)
          ) :
        train_dataset = StereoLowDataset(**asdict(data_cfg))
        train_video_view = StereoLowDatasetVideoView(train_dataset)
        if load_val:
            val_img_dataset = (
                StereoLowDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    elif (isinstance(data_cfg, StereoHighDataConfig)) :
        train_dataset = StereoHighDataset(**asdict(data_cfg))
        train_video_view = StereoHighDatasetVideoView(train_dataset)
        if load_val:
            val_img_dataset = (
                StereoHighDataset(
                    **asdict(replace(data_cfg, split="val", load_from_cache=True))
                )
                if train_dataset.has_validation
                else None
            )
    else:
        raise ValueError(f"Unknown data config: {data_cfg}")
    return train_dataset, train_video_view, val_img_dataset, val_kpt_dataset
