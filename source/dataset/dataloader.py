import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F


def _set_training_steps(cfg: DictConfig, train_length: int) -> None:
    with open_dict(cfg):
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs


def _make_dataloaders(cfg: DictConfig,
                      dataset: utils.TensorDataset,
                      train_index: np.ndarray,
                      val_index: np.ndarray,
                      test_index: np.ndarray) -> List[utils.DataLoader]:
    train_dataset = utils.Subset(dataset, train_index.tolist())
    val_dataset = utils.Subset(dataset, val_index.tolist())
    test_dataset = utils.Subset(dataset, test_index.tolist())

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    _set_training_steps(cfg, train_length)

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    _set_training_steps(cfg, train_length)

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    for train_index, test_valid_index in split.split(final_timeseires, stratified):
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
        stratified = stratified[test_valid_index]

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    for valid_index, test_index in split2.split(final_timeseires_val_test, stratified):
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_kfold_dataloaders(cfg: DictConfig,
                                      final_timeseires: torch.tensor,
                                      final_pearson: torch.tensor,
                                      labels: torch.tensor,
                                      stratified: np.array) -> List[List[utils.DataLoader]]:
    labels = labels.to(torch.int64)
    labels_one_hot = F.one_hot(labels)
    labels_np = labels.cpu().numpy()
    length = final_timeseires.shape[0]
    n_splits = int(cfg.training.get("num_folds", 5))
    validation_set = float(cfg.training.get("validation_set", cfg.dataset.val_set))
    random_state = int(cfg.training.get("split_seed", 42))
    train_percentage = float(cfg.datasz.percentage)

    dataset = utils.TensorDataset(
        final_timeseires,
        final_pearson,
        labels_one_hot
    )

    fold_dataloaders = []
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_val_index, test_index) in enumerate(
            splitter.split(np.zeros(length), labels_np)):
        train_val_labels = labels_np[train_val_index]
        val_size = max(1, int(round(len(train_val_index) * validation_set)))
        val_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size, random_state=random_state + fold_idx)
        train_relative, val_relative = next(
            val_splitter.split(np.zeros(len(train_val_index)), train_val_labels))

        train_index = train_val_index[train_relative]
        val_index = train_val_index[val_relative]

        if train_percentage < 1.0:
            train_size = max(1, int(round(len(train_index) * train_percentage)))
            train_subsetter = StratifiedShuffleSplit(
                n_splits=1, train_size=train_size, random_state=random_state + 100 + fold_idx)
            train_relative, _ = next(
                train_subsetter.split(np.zeros(len(train_index)), labels_np[train_index]))
            train_index = train_index[train_relative]

        if fold_idx == 0:
            _set_training_steps(cfg, len(train_index))

        fold_dataloaders.append(
            _make_dataloaders(cfg, dataset, train_index, val_index, test_index))

    return fold_dataloaders
