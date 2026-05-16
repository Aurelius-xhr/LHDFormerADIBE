from omegaconf import DictConfig, open_dict
from .abide import load_abide_data
from .dataloader import init_dataloader, init_stratified_dataloader, init_stratified_kfold_dataloaders
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abide']

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    if cfg.dataset.stratified and int(cfg.training.get("num_folds", 1)) > 1:
        dataloaders = init_stratified_kfold_dataloaders(cfg, *datasets)
    else:
        dataloaders = init_stratified_dataloader(cfg, *datasets) \
            if cfg.dataset.stratified \
            else init_dataloader(cfg, *datasets)

    return dataloaders
