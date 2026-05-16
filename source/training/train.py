from source.utils import accuracy, TotalMeter, count_params, isfloat
import copy
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import pickle
from ..utils.device import device_set
device = device_set()

class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        self.save_attn_weights = cfg.save_attn_weights
        self.save_test_attn_weights = cfg.save_test_attn_weights

        self.init_meters()
        self.mseLoss = torch.nn.MSELoss()
        self.l1Loss = torch.nn.L1Loss()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()

        for time_series, node_feature, label in self.train_dataloader:
            label = label.float()
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)

            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)

            predict, _ = self.model(time_series, node_feature)

            loss = self.loss_fn(predict, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(predict, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        with torch.no_grad():
            for time_series, node_feature, label in dataloader:
                time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)
                output, loss_pool = self.model(time_series, node_feature)

                label = label.float()

                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(
                    loss.item(), label.shape[0])
                top1 = accuracy(output, label[:, 1])[0]
                acc_meter.update_with_weight(top1, label.shape[0])
                result += F.softmax(output, dim=1)[:, 1].tolist()
                labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)
        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        report = classification_report(
            labels, result, output_dict=True, zero_division=0)

        recall = [0, 0]
        for k in report:
            if isfloat(k):
                recall[int(float(k))] = report[k]['recall']
        return [auc] + list(metric) + recall

    def save_attention_weights(self):
        attn_weights = []
        labels = []
        assign_matrices = []
        attn_weights_local = []
        self.model.eval()
        for time_series, node_feature, label in self.train_dataloader:
            time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)
            prediction, _ = self.model(time_series, node_feature)

            assignMat = self.model.get_assign_mat()
            if assignMat is not None:
                assign_np = assignMat.detach().cpu().numpy()
                assign_matrices.append(assign_np)

            attn = self.model.get_attention_weights()
            attn_np = attn[0].detach().cpu().numpy() if attn is not None else None
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            if attn_np is not None:
                attn_weights.append(attn_np)


        for time_series, node_feature, label in self.val_dataloader:
            time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)
            prediction, _ = self.model(time_series, node_feature)

            assignMat = self.model.get_assign_mat()
            if assignMat is not None:
                assign_np = assignMat.detach().cpu().numpy()
                assign_matrices.append(assign_np)


            attn = self.model.get_attention_weights()
            attn_np = attn[0].detach().cpu().numpy() if attn is not None else None
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            if attn_np is not None:
                attn_weights.append(attn_np)

        if self.save_test_attn_weights:
            attn_weights_test = []
            labels_test = []
            assign_matrices_test = []

        for time_series, node_feature, label in self.test_dataloader:
            time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)
            prediction, _ = self.model(time_series, node_feature)

            assignMat = self.model.get_assign_mat()
            assign_np = None
            if assignMat is not None:
                assign_np = assignMat.detach().cpu().numpy()
                assign_matrices.append(assign_np)


            attn = self.model.get_attention_weights()
            attn_np = attn[0].detach().cpu().numpy() if attn is not None else None
            label_np = label.detach().cpu().numpy()
            labels.append(label_np)
            if attn_np is not None:
                attn_weights.append(attn_np)

            if self.save_test_attn_weights:
                if assign_np is not None:
                    assign_matrices_test.append(assign_np)
                labels_test.append(label_np)
                if attn_np is not None:
                    attn_weights_test.append(attn_np)


        np.save(self.save_path/f"attnWeights.npy", attn_weights, allow_pickle=True)
        np.save(self.save_path/f"labels.npy", labels, allow_pickle=True)
        # np.save(self.save_path/f"assign_matrices.npy", assign_matrices, allow_pickle=True)

        if self.save_test_attn_weights:
            np.save(self.save_path/f"attnWeights_test.npy", attn_weights_test, allow_pickle=True)
            np.save(self.save_path/f"labels_test.npy", labels_test, allow_pickle=True)
            np.save(self.save_path/f"assign_matrices_test.npy", assign_matrices_test, allow_pickle=True)

    def generate_save_learnable_matrix(self):

        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.to(device), node_feature.to(device), label.to(device)
            learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        self.current_step = 0
        best_val_AUC = -float("inf")
        best_val_acc = 0
        best_epoch = -1
        best_state_dict = None
        epochs_without_improvement = 0
        early_stopping_patience = int(
            self.config.training.get("early_stopping_patience", 20))
        early_stopping_delta = float(
            self.config.training.get("early_stopping_delta", 0.0))
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}',
                f'Val AUC:{val_result[0]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.5f}'
            ]))

            wandb.log({
                "Train Loss": self.train_loss.avg,
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.avg,
                "Val AUC": val_result[0],
                "LR": self.lr_schedulers[0].lr,
                "Epoch": epoch,
            })

            if val_result[0] > best_val_AUC + early_stopping_delta:
                best_epoch = epoch
                best_val_AUC = val_result[0]
                best_val_acc = self.val_accuracy.avg
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                wandb.run.summary["Best Val AUC"] = val_result[0]
                wandb.run.summary["Best Val Accuracy"] = self.val_accuracy.avg
                wandb.run.summary["Best Epoch"] = epoch
            else:
                epochs_without_improvement += 1

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Train Accuracy": self.train_accuracy.avg,
                "Val AUC": val_result[0],
                "Val Loss": self.val_loss.avg,
                "Val Accuracy": self.val_accuracy.avg,
            })

            if early_stopping_patience > 0 and \
                    epochs_without_improvement >= early_stopping_patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, Best Val AUC: {best_val_AUC:.4f}")
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.test_loss.reset()
        self.test_accuracy.reset()
        test_result = self.test_per_epoch(self.test_dataloader,
                                          self.test_loss, self.test_accuracy)
        self.logger.info(" | ".join([
            f'Final Test Accuracy:{self.test_accuracy.avg: .3f}%',
            f'Final Test AUC:{test_result[0]:.4f}',
            f'Final Test Sen:{test_result[-1]:.4f}',
            f'Final Test Spe:{test_result[-2]:.4f}',
            f'Best Epoch:{best_epoch}',
        ]))
        wandb.log({
            "Final Test Loss": self.test_loss.avg,
            "Final Test Accuracy": self.test_accuracy.avg,
            "Final Test AUC": test_result[0],
            "Final Test Sensitivity": test_result[-1],
            "Final Test Specificity": test_result[-2],
        })
        wandb.run.summary["Final Test Accuracy"] = self.test_accuracy.avg
        wandb.run.summary["Final Test AUC"] = test_result[0]
        wandb.run.summary["Final Test Sensitivity"] = test_result[-1]
        wandb.run.summary["Final Test Specificity"] = test_result[-2]

        if self.save_attn_weights:
            self.save_attention_weights()

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()

        self.save_result(training_process)
        return [self.test_accuracy.avg, test_result[0], test_result[-1], test_result[-2]]
