from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split

from gnn.mol2graph import Mol2Graph


CORES = 2


class GraphDataset:
    def __init__(
        self,
        smiles: np.ndarray,
        y: np.ndarray,
        mol2graph: Mol2Graph,
        batch_size: int = 1,
        dataset_ratio: list[float] = [0.8, 0.1, 0.1],
        random_state: int = 42,
    ):
        if sum(dataset_ratio) != 1.0:
            raise RuntimeError("Make sure the sum of the ratios is 1.")
        # ========== 変数の設定 ==========
        self.smiles = smiles
        self.y = y
        self.mol2graph = mol2graph
        self.batch_size = batch_size
        self.dataset_ratio = dataset_ratio
        self.random_state = random_state
        self.n_node_features, self.n_edge_features = self.mol2graph.get_base_graph_features()
        # ========= データセットの分割 ==========
        X_train, X_test, y_train, y_test = train_test_split(
            self.smiles,
            self.y,
            test_size=1 - self.dataset_ratio[0],
            shuffle=True,
            random_state=self.random_state,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test,
            y_test,
            test_size=(
                self.dataset_ratio[2]
                / (self.dataset_ratio[2] + self.dataset_ratio[1])
            ),
            shuffle=True,
            random_state=self.random_state,
        )
        self.train_dataset = [X_train, y_train]
        self.valid_dataset = [X_valid, y_valid]
        self.test_dataset = [X_test, y_test]

    def split_dataset(
        self,
        validation_method: Literal["holdout", "cv"] = "holdout"
    ):
        datasets = dict()
        # ========= バリデーション方法の設定 ==========
        # ホールドアウト法によるデータ分割
        if validation_method == "holdout":
            splitted_datasets = {
                "train": self.train_dataset,
                "valid": self.valid_dataset,
                "test": self.test_dataset,
            }
        else:
            # k-fold法によるデータ分割
            X_train = np.concatenate(
                [
                    self.train_dataset[0],
                    self.valid_dataset[0]
                ], axis=0
            )
            y_train = np.concatenate(
                [
                    self.train_dataset[1],
                    self.valid_dataset[1]
                ], axis=0
            )
            splitted_datasets = {
                "train": [X_train, y_train],
                "test": [self.test_dataset[0], self.test_dataset[1]]
            }
        # ======== グラフデータセットの作成 ==========
        for phase, [X, y] in splitted_datasets.items():
            datasets[phase] = self.mol2graph.get_graph_vectors(
                X, y, self.n_node_features, self.n_edge_features
            )
        return datasets
