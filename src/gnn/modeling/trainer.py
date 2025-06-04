import copy
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm
from torch.utils.data.dataset import Subset
from torchmetrics import (
    MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
)
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from gnn.model import MolecularGNN


class EarlyStopping:
    counter = 0
    best_score: float | None = None
    early_stop = False
    best_val_loss = np.inf

    def __init__(self, patience=100, model_dir_path=Path(".")):
        self.patience = patience
        self.model_dir_path = model_dir_path

    def __call__(
        self,
        val_loss: float,
        model: MolecularGNN,
        save_filename: str
    ):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self._checkpoint(val_loss, model, save_filename)

        elif score > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._checkpoint(val_loss, model, save_filename)
            self.counter = 0

    def _checkpoint(
        self,
        val_loss: float,
        model: MolecularGNN,
        save_filename: str
    ):
        torch.save(
            model.state_dict(), self.model_dir_path.joinpath(save_filename)
        )
        self.best_val_loss = val_loss


class Trainer:
    loss = 0.0
    history: defaultdict[str, list] = defaultdict(list)

    def __init__(
        self,
        output_dir: Path,
        learning_rate: float,
        scheduler=None,
        n_epochs: int = 100,
        batch_size: int = 1,
        early_stopping: Optional[EarlyStopping] = None,
        device="cpu",
    ):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.device = device

    def _epoch_train(
        self,
        model: MolecularGNN,
        criterion: nn.MSELoss,
        optimizer: optim.AdamW,
        dataloader: DataLoader,
        metrics: MetricCollection,
    ):
        epoch_loss = 0.0
        epoch_y_true = torch.Tensor()
        epoch_y_pred = torch.Tensor()
        model.train()

        for data in dataloader:
            data = data.to(self.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = torch.sqrt(criterion(outputs, data.y))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if self.scheduler is not None:
                self.scheduler.step()
            epoch_y_true = torch.cat((epoch_y_true.cpu(), data.y.cpu()))
            epoch_y_pred = torch.cat((epoch_y_pred.cpu(), outputs.cpu()))
        # epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_loss = epoch_loss / len(dataloader)
        all_metrics = metrics(epoch_y_true, epoch_y_pred)
        return epoch_loss, all_metrics

    def _epoch_valid(
        self,
        model: MolecularGNN,
        criterion: nn.MSELoss,
        dataloader: DataLoader,
        metrics: MetricCollection
    ):
        epoch_loss = 0.0
        epoch_y_true = torch.Tensor()
        epoch_y_pred = torch.Tensor()

        model.eval()
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                outputs = model(data)
                loss = torch.sqrt(criterion(outputs, data.y))
                epoch_loss += loss.item()
                epoch_y_true = torch.cat((epoch_y_true.cpu(), data.y.cpu()))
                epoch_y_pred = torch.cat((epoch_y_pred.cpu(), outputs.cpu()))
        # epoch_loss = epoch_loss / len(dataloader.dataset)
        epoch_loss = epoch_loss / len(dataloader)
        all_metrics = metrics(epoch_y_true, epoch_y_pred)
        return epoch_loss, all_metrics

    def _refresh_metrics(self):
        self.history = defaultdict(list)

    def _save_metrics(self, filename: str):
        metrics_dirpath = self.output_dir.joinpath(filename)
        metrics_df = pd.DataFrame.from_dict(self.history)
        metrics_df.index = range(1, len(metrics_df) + 1)
        metrics_df.reset_index().rename(columns={"index": "epoch"})
        metrics_df.to_csv(metrics_dirpath)

    def _initialize_early_stopping(self):
        if self.early_stopping is not None:
            self.early_stopping.counter = 0
            self.early_stopping.best_score = None
            self.early_stopping.early_stop = False
            self.early_stopping.best_val_loss = np.Inf


class CrossValidationTrainer(Trainer):
    def __init__(
        self,
        output_dir: Path,
        learning_rate: float,
        scheduler=None,
        n_epochs: int = 100,
        batch_size: int = 1,
        cv_n_splits: int = 5,
        early_stopping: Optional[EarlyStopping] = None,
        device="cpu",
    ):
        super().__init__(
            output_dir,
            learning_rate,
            scheduler,
            n_epochs,
            batch_size,
            early_stopping,
            device,
        )
        self.cv_n_splits = cv_n_splits
        self.kf = KFold(
            n_splits=self.cv_n_splits,
            shuffle=True,
            random_state=42,
        )

    def fit(self, model: MolecularGNN, graph_datasets: dict):
        metrics = MetricCollection(
            metrics={
                "RMSE": MeanSquaredError(squared=False, num_outputs=1),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(
                    model.output_layer.out_features,
                    multioutput="uniform_average"
                ),
            }
        )

        for fold, (train_idx, valid_idx) in enumerate(
            self.kf.split(graph_datasets["train"])
        ):
            fold_model = copy.deepcopy(model).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                fold_model.parameters(),
                lr=self.learning_rate
            )
            self._initialize_early_stopping()
            cv_loss = 0.0
            train_dataset = Subset(graph_datasets["train"], train_idx)
            valid_dataset = Subset(
                graph_datasets["train"], valid_idx
            )
            train_dataloader = DataLoader(
                train_dataset,
                self.batch_size,
                shuffle=True,
                num_workers=2,
                drop_last=True,
                persistent_workers=True,
            )
            valid_dataloader = DataLoader(
                valid_dataset,
                self.batch_size,
                shuffle=False,
                num_workers=2,
                drop_last=True,
                persistent_workers=True,
            )
            with tqdm(range(1, self.n_epochs + 1)) as progress_bar:
                for _epoch in progress_bar:
                    progress_bar.set_description_str(f"Fold: {fold}")
                    epoch_train_loss, train_metrics = self._epoch_train(
                        fold_model,
                        criterion,
                        optimizer,
                        train_dataloader,
                        metrics
                    )
                    self.history["train_RMSE"].append(train_metrics["RMSE"].item())
                    self.history["train_MAE"].append(train_metrics["MAE"].item())
                    self.history["train_R2"].append(train_metrics["R2"].item())

                    epoch_valid_loss, valid_metrics = self._epoch_valid(
                        fold_model, criterion, valid_dataloader, metrics
                    )
                    self.history["valid_RMSE"].append(valid_metrics["RMSE"].item())
                    self.history["valid_MAE"].append(valid_metrics["MAE"].item())
                    self.history["valid_R2"].append(valid_metrics["R2"].item())
                    progress_bar.set_postfix_str(
                        f"loss={epoch_train_loss:.4f} "
                        + f"valid_loss={epoch_valid_loss:.4f} "
                        + f"valid_r2={valid_metrics['R2'].item():.4f}"
                    )

                    if self.early_stopping is not None:
                        self.early_stopping(
                            epoch_valid_loss,
                            fold_model,
                            f"{fold}_fold_model_params.pth"
                        )
                        if self.early_stopping.early_stop:
                            cv_loss = self.early_stopping.best_score
                            break
                    else:
                        cv_loss = epoch_valid_loss
                self.loss += cv_loss
            filename = f"fold{fold}_metrics.csv"
            self._save_metrics(filename)
            self._refresh_metrics()
            self.loss /= self.cv_n_splits


class HoldOutTrainer(Trainer):
    def __init__(
        self,
        output_dir: Path,
        learning_rate: float,
        scheduler=None,
        n_epochs: int = 100,
        batch_size: int = 1,
        early_stopping: None | EarlyStopping = None,
        device="cpu",
    ):
        super().__init__(
            output_dir,
            learning_rate,
            scheduler,
            n_epochs,
            batch_size,
            early_stopping,
            device,
        )

    def fit(self, model: MolecularGNN, graph_datasets):
        model = model.to(self.device)
        metrics = MetricCollection(
            metrics={
                "RMSE": MeanSquaredError(squared=False, num_outputs=1),
                "MAE": MeanAbsoluteError(),
                "R2": R2Score(
                    model.output_layer.out_features,
                    multioutput="uniform_average"
                ),
            }
        )
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        train_dataloader = DataLoader(
            graph_datasets["train"],
            self.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            persistent_workers=True,
        )
        valid_dataloader = DataLoader(
            graph_datasets["valid"],
            self.batch_size,
            shuffle=False,
            num_workers=2,
            drop_last=True,
            persistent_workers=True,
        )
        with tqdm(range(1, self.n_epochs + 1)) as progress_bar:
            for _epoch in progress_bar:
                train_loss, train_metrics = self._epoch_train(
                    model, criterion, optimizer, train_dataloader, metrics
                )
                self.history["train_RMSE"].append(train_metrics["RMSE"].item())
                self.history["train_MAE"].append(train_metrics["MAE"].item())
                self.history["train_R2"].append(train_metrics["R2"].item())
                progress_bar.set_postfix_str(f"loss={train_loss:.4f}")

                valid_loss, valid_metrics = self._epoch_valid(
                    model, criterion, valid_dataloader, metrics
                )
                self.history["valid_RMSE"].append(valid_metrics["RMSE"].item())
                self.history["valid_MAE"].append(valid_metrics["MAE"].item())
                self.history["valid_R2"].append(valid_metrics["R2"].item())
                progress_bar.set_postfix_str(
                    f"loss={train_loss:.4f} "
                    + f"valid_loss={valid_loss:.4f} "
                    + f"valid_r2={valid_metrics['R2'].item():.4f}"
                )
                if self.early_stopping is not None:
                    self.early_stopping(valid_loss, model, "model_params.pth")
                    if self.early_stopping.early_stop:
                        valid_loss = self.early_stopping.best_score
                        break
        filename = "metrics.csv"
        self._save_metrics(filename)
        self.loss = valid_loss
