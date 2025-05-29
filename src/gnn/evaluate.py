from logging import Logger
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    root_mean_squared_error
)
from torch_geometric.loader import DataLoader

from gnn import plot
from gnn.model import MolecularGNN


np.set_printoptions(precision=3)


def evaluate_model(
    model: MolecularGNN,
    logger: Logger,
    output_dir: Path,
    graph_datasets: dict,
    batch_size: int,
    device="cpu",
):
    """モデルの評価を行う関数

    Parameters
    ----------
    model : MolecularGNN
        評価を行うモデル
    logger : Logger
        ロガー
    output_dir : Path
        出力先のディレクトリ
    graph_datasets : dict
        データセットの辞書
        キーは "train", "valid", "test" で、値は DataLoader
    batch_size : int
        評価を行う際のバッチサイズ
    device : str, optional
        計算デバイス, by default "cpu"
    """
    # Output用のディレクトリを作成
    img_dir = output_dir.joinpath("figures")
    img_dir.mkdir(exist_ok=True)
    model_dir = output_dir.joinpath("model")
    result = dict()
    for phase, graph_dataset in graph_datasets.items():
        dataloader = DataLoader(
            graph_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        result[phase] = compute_metrics(
            model, model_dir, dataloader, device
        )
        logger.info(f"For the {phase} set:")
        logger.info(
            f"MAE: {result[phase]['MAE']:.4f} "
            + f"RMSE: {result[phase]['RMSE']:.4f} "
            + f"R^2: {result[phase]['R2']:.4f}"
        )

    plot.plot_obserbations_vs_predictions(result, img_dir=img_dir)


def compute_metrics(
    model: MolecularGNN,
    output_dir: Path,
    dataloader: DataLoader,
    device="cpu",
):
    """_summary_

    Parameters
    ----------
    model : MolecularGNN
        _description_
    output_dir : Path
        _description_
    dataloader : DataLoader
        _description_
    device : str, optional
        _description_, by default "cpu"

    Returns
    -------
    _type_
        _description_
    """
    pth_path_list = list(output_dir.glob("*.pth"))
    y_list, y_pred_list = [], []

    for dataset in dataloader:
        dataset = dataset.to(device)
        y = dataset.y.cpu().detach().numpy()
        y_pred = np.zeros_like(y)
        for i, pth_path in enumerate(pth_path_list):
            model.load_state_dict(torch.load(pth_path))
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                y_pred_tmp = model.forward(dataset)
            y_pred_tmp = y_pred_tmp.cpu().detach().numpy()
            if i == 0:
                y_pred = y_pred_tmp.copy()
            else:
                y_pred = np.concatenate([y_pred, y_pred_tmp], axis=1)
        if len(pth_path_list) != 1:
            y_pred = y_pred.mean(axis=1).reshape(-1, 1)
        y_list.extend(list(y))
        y_pred_list.extend(list(y_pred))
    mae = float(mean_absolute_error(y_list, y_pred_list))
    rmse = float(root_mean_squared_error(y_list, y_pred_list))
    r2 = float(r2_score(y_list, y_pred_list))
    result = {
        "y": y_list,
        "y_pred": y_pred_list,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    return result
