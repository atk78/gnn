import shutil
import os
import platform
import warnings
import random
from logging import Logger
from pathlib import Path
from typing import Literal, Union

import yaml
import polars as pl
import numpy as np
import optuna
import torch

from gnn.model import MolecularGNN
from gnn.mol2graph import Mol2Graph
from gnn.dataset import GraphDataset
from gnn.modeling.trainer import HoldOutTrainer, CrossValidationTrainer, EarlyStopping
from gnn.modeling.evaluate import evaluate_model
from gnn import utils


warnings.simplefilter("ignore")


class BayOptLoss:
    loss = None
    r2 = None
    number = 0


def seed_everything(seed: int):
    """乱数のシードを固定する関数

    Parameters
    ----------
    seed : int
        乱数のシード
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def bayopt_hparams(
    output_dir: Path,
    bayopt_datasets: dict[str, GraphDataset],
    logger: Logger,
    n_features: int,
    gnn_type: Literal["GAT", "GCN"],
    bayopt_bounds: dict,
    validation_method: Literal["holdout", "cv"] = "holdout",
    cv_n_splits=5,
    n_epochs=10,
    batch_size=32,
    n_trials=10,
    num_of_outputs=1,
    seed=42,
    device="cpu"
):
    """Optunaを用いてハイパーパラメータの探索を行う関数

    Parameters
    ----------
    output_dir : Path
        出力ディレクトリのPathオブジェクト
    bayopt_datasets : dict[str, GraphDataset]
        学習用のデータセット
    logger : Logger
        ロガー
    n_features : int
        GNNのノード特徴量の次元数
    gnn_type : Literal['GAT', 'GCN']
        GNNの種類。'GAT'または'GCN'を指定。
    bayopt_bounds : dict
        ハイパーパラメータの探索範囲。
    validation_method : Literal['holdout', 'cv'], optional
        validationの方法。
    cv_n_splits : int, optional
        validation_method = "cv"の場合、クロスバリデーションの分割数, by default 5
    n_epochs : int, optional
        エポック数, by default 10
    batch_size : int, optional
        バッチサイズ, by default 32
    n_trials : int, optional
        ハイパーパラメータ探索の試行回数, by default 10
    num_of_outputs : int, optional
        出力の数, by default 1
    seed : int, optional
        乱数シード, by default 42
    device : str, optional
        使用するデバイス, by default "cpu"

    Returns
    -------
    best_hparameters : dict
        ベイズ最適化によって導出されたハイパーパラメータの辞書
    """
    bayopt_dir = output_dir.joinpath("bayes_opt")
    if bayopt_dir.exists():
        shutil.rmtree(bayopt_dir)
    bayopt_dir.mkdir()
    optuna.logging.enable_propagation()
    # Optunaの学習用関数を内部に作成
    def _objective(trial: optuna.trial.Trial):
        lr = trial.suggest_float(
            "learning_rate",
            float(bayopt_bounds["learning_rate"][0]),
            float(bayopt_bounds["learning_rate"][1]),
            log=True,
        )

        bayopt_model = make_opt_model(
            bayopt_bounds,
            n_features,
            gnn_type,
            trial,
            num_of_outputs
        )
        # criterion = nn.MSELoss()
        # optimizer = optim.AdamW(opt_model.parameters(), lr=lr)
        trial_path = bayopt_dir.joinpath(f"trial_{trial.number}")
        trial_path.mkdir(exist_ok=True)

        bayopt_trainer: Union[HoldOutTrainer, CrossValidationTrainer]
        if validation_method == "cv":
            bayopt_trainer = CrossValidationTrainer(
                trial_path,
                learning_rate=lr,
                scheduler=None,
                n_epochs=n_epochs,
                batch_size=batch_size,
                cv_n_splits=cv_n_splits,
                early_stopping=None,
                device=device,
            )
        else:
            bayopt_trainer = HoldOutTrainer(
                trial_path,
                learning_rate=lr,
                scheduler=None,
                n_epochs=n_epochs,
                batch_size=batch_size,
                early_stopping=None,
                device=device,
            )
        # モデルの学習
        bayopt_trainer.fit(bayopt_model, bayopt_datasets)
        if BayOptLoss.loss is None:
            BayOptLoss.loss = bayopt_trainer.loss
        else:
            if BayOptLoss.loss > bayopt_trainer.loss:
                BayOptLoss.loss = bayopt_trainer.loss
                BayOptLoss.number = trial.number
        return bayopt_trainer.loss

    # ハイパーパラメータの探索の開始
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=1)
    # 探索のうち、一番損失が少なかった条件でのハイパーパラメータを保存
    trial = study.best_trial
    logger.info(
        f"Best Trial: {trial.number} with RMSE value: {trial.value}"
    )
    best_hparams = {
        "n_features": int(n_features),
        "n_conv_hidden_layer": trial.params["n_conv_hidden_layer"],
        "n_dense_hidden_layer": trial.params["n_dense_hidden_layer"],
        "graph_dim": 2 ** trial.params["n_graph_dim"],
        "dense_dim": 2 ** trial.params["n_dense_dim"],
        "drop_rate": trial.params["drop_rate"],
        "gnn_type": gnn_type,
        "learning_rate": trial.params["learning_rate"],
        "num_of_outputs": num_of_outputs,
    }
    return best_hparams


def make_opt_model(
    bayopt_bounds: dict,
    n_features: int,
    gnn_type: str,
    trial: optuna.trial.Trial,
    num_of_outputs=1,
):
    """OptunaのトライアルからGNNモデルを生成する関数

    Parameters
    ----------
    bayopt_bounds : dict
        ハイパーパラメータの探索範囲。
    n_features : int
        GNNのノード特徴量の次元数
    gnn_type : str
        GNNの種類。'GAT'または'GCN'を指定。
    trial : optuna.trial.Trial
        Optunaのトライアルオブジェクト
    num_of_outputs : int, optional
        出力の数, by default 1

    Returns
    -------
    MolecularGNN
        生成されたGNNモデル
    """
    # ハイパーパラメータの探索範囲から値を取得
    n_conv_hidden_layer = trial.suggest_int(
        "n_conv_hidden_layer",
        bayopt_bounds["n_conv_hidden_layer"][0],
        bayopt_bounds["n_conv_hidden_layer"][1],
    )
    n_dense_hidden_layer = trial.suggest_int(
        "n_dense_hidden_layer",
        bayopt_bounds["n_dense_hidden_layer"][0],
        bayopt_bounds["n_dense_hidden_layer"][1],
    )
    n_graph_dim = trial.suggest_int(
        "n_graph_dim",
        bayopt_bounds["graph_dim"][0],
        bayopt_bounds["graph_dim"][1],
    )
    n_dense_dim = trial.suggest_int(
        "n_dense_dim",
        bayopt_bounds["dense_dim"][0],
        bayopt_bounds["dense_dim"][1],
    )
    drop_rate = trial.suggest_discrete_uniform(
        "drop_rate",
        bayopt_bounds["drop_rate"][0],
        bayopt_bounds["drop_rate"][1],
        bayopt_bounds["drop_rate"][2]
    )
    # 探索用に生成されたハイパーパラメータを用いてモデルを生成
    opt_model = MolecularGNN(
        n_features,
        n_conv_hidden_layer,
        n_dense_hidden_layer,
        2**n_graph_dim,
        2**n_dense_dim,
        drop_rate,
        gnn_type,
        num_of_outputs
    )
    return opt_model


def training_model(
    model: MolecularGNN,
    output_dir: Path,
    datasets: dict[str, GraphDataset],
    learning_rate: float,
    n_epochs=100,
    batch_size=32,
    early_stopping_patience=0,
    device="cpu",
):
    """GNNモデルの本学習を行う関数

    Parameters
    ----------
    model : MolecularGNN
        学習するGNNモデル
    output_dir : Path
        出力ディレクトリのPathオブジェクト
    datasets : dict[str, GraphDataset]
        学習用のデータセット
    learning_rate : float
        学習率
    n_epochs : int, optional
        エポック数, by default 100
    batch_size : int, optional
        バッチサイズ, by default 32
    early_stopping_patience : int, optional
        Early Stoppingのエポック数, by default 0
    device : str, optional
        使用するデバイス, by default "cpu"

    Returns
    -------
    model : MolecularGNN
        学習後のGNNモデル
    """
    training_dir = output_dir.joinpath("training")
    training_dir.mkdir()
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            early_stopping_patience,
            output_dir.joinpath("model"),
        )
    else:
        early_stopping = None
    trainer = HoldOutTrainer(
        training_dir,
        learning_rate,
        scheduler=None,
        n_epochs=n_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        device=device
    )
    trainer.fit(model, datasets)
    return model


def copy_model_params(output_dir: Path):
    """学習後のモデルパラメータを指定のディレクトリにコピーする関数

    Parameters
    ----------
    output_dir : Path
        学習結果の出力ディレクトリのPathオブジェクト
    """
    output_dir_name = output_dir.name
    parent_dir = output_dir.parent.parent
    model_dir = parent_dir.joinpath("model").joinpath(output_dir_name)
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        output_dir.joinpath("model/all_params.yml"),
        model_dir.joinpath("all_params.yml")
    )
    shutil.copy(
        output_dir.joinpath("model/model_params.pth"),
        model_dir.joinpath("model_params.pth")
    )


def run(
    config_filepath: str,
    # n_gpus=1,
    n_conv_hidden_ref=1,
    n_dense_hidden_ref=1,
    graph_dim_ref=64,
    dense_dim_ref=64,
    drop_rate_ref=0.1,
    lr_ref=1e-3,
):
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    # *****************************************
    # 変数の設定
    # *****************************************
    # ハイパーパラメータの最適化条件設定
    bayopt_on = config["bayopt_hparams"]["bayopt_on"]
    if bayopt_on:
        bayopt_bounds = config["bayopt_bounds"]
        bayopt_validation_method = config["bayopt_hparams"]["validation_method"]
        if bayopt_validation_method not in ["holdout", "cv"]:
            raise ValueError(
                "bayopt_validation_method must be 'holdout' or 'cv'."
            )
        if bayopt_validation_method == "cv":
            bayopt_cv_n_splits = config["bayopt_hparams"]["cv_n_splits"]
        bayopt_n_epochs = config["bayopt_hparams"]["n_epochs"]
        bayopt_n_trials = config["bayopt_hparams"]["n_trials"]
        bayopt_batch_size = config["bayopt_hparams"]["batch_size"]

    # 学習の条件設定
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    early_stopping_patience = config["train"]["early_stopping_patience"]
    tf16 = config["tf16"]
    seed = config["seed"]

    # データセットの設定
    dataset_filepath = config["dataset"]["filepath"]
    output_dir = config["dataset"]["output_dir"]
    smiles_col_name = config["dataset"]["smiles_col_name"]
    prop_col_name = config["dataset"]["prop_col_name"]
    dataset_ratio = config["dataset"]["dataset_ratio"]

    # モデルの設定
    computable_atoms = config["computable_atoms"]
    chirality = config["chirality"]
    stereochemistry = config["stereochemistry"]
    gnn_type = config["gnn_type"]

    if type(prop_col_name) is str:
        prop_col_name = [prop_col_name]

    # *****************************************
    # データセットの読み込み
    # *****************************************
    num_of_outputs = len(prop_col_name)
    dataset = pl.read_csv(dataset_filepath)
    dataset = dataset.select(smiles_col_name, *prop_col_name)
    print(dataset.head())

    # *****************************************
    # 出力ディレクトリの設定
    # *****************************************
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir()
    model_dir = output_dir.joinpath("model")
    model_dir.mkdir()

    logger = utils.log_setup(log_dir, "training", verbose=True)
    logger.info(f"OS: {platform.system()}")

    # *****************************************
    # 計算精度の設定
    # *****************************************
    if tf16:
        precision = "16"
        logger.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32"
        logger.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("high")
    seed_everything(seed)

    # *****************************************
    # データセットの前処理
    # *****************************************
    if dataset[smiles_col_name][0].count("*") == 0:
        poly_flag = False
    else:
        poly_flag = True
    device = (
        torch.device("cuda:0") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # データのグラフ化処理
    mol2graph = Mol2Graph(
        computable_atoms, poly_flag, chirality, stereochemistry
    )
    graph_dataset = GraphDataset(
        smiles=dataset[smiles_col_name].to_numpy(),
        y=dataset[*prop_col_name].to_numpy(),
        mol2graph=mol2graph,
        batch_size=batch_size,
        dataset_ratio=dataset_ratio,
        random_state=seed,
    )
    training_datasets = graph_dataset.split_dataset("holdout")
    # グラフのノード特徴量の次元数
    n_features = len(training_datasets["train"][0]["x"][0])
    # *****************************************
    # ハイパーパラメータの最適化
    # *****************************************
    if bayopt_on:
        logger.info(f"Dava validation method: {bayopt_validation_method}.")
        if bayopt_validation_method == "cv":
            logger.info(f"Num of fold: {bayopt_cv_n_splits}")
            bayopt_datasets = graph_dataset.split_dataset("cv")
        else:
            bayopt_datasets = graph_dataset.split_dataset("holdout")
        # optuna.logger.disable_default_handler()
        best_hparams = bayopt_hparams(
            output_dir,
            bayopt_datasets,
            logger,
            n_features,
            gnn_type,
            bayopt_bounds,
            bayopt_validation_method,
            bayopt_cv_n_splits,
            bayopt_n_epochs,
            bayopt_batch_size,
            bayopt_n_trials,
            num_of_outputs,
            seed,
            device
        )
    else:
        best_hparams = {
            "n_features": n_features,
            "n_conv_hidden_layer": n_conv_hidden_ref,
            "n_dense_hidden_layer": n_dense_hidden_ref,
            "graph_dim": graph_dim_ref,
            "dense_dim": dense_dim_ref,
            "drop_rate": drop_rate_ref,
            "gnn_type": gnn_type,
            "learning_rate": lr_ref,
            "num_of_outputs": num_of_outputs,
        }
    logger.info("Best Params")
    logger.info(f"GNN Type            |{best_hparams['gnn_type']}")
    logger.info(f"Conv hidden layers  |{best_hparams['n_conv_hidden_layer']}")
    logger.info(f"Dense hidden layers |{best_hparams['n_dense_hidden_layer']}")
    logger.info(f"Graph dim           |{best_hparams['graph_dim']}")
    logger.info(f"Dense dim           |{best_hparams['dense_dim']}")
    logger.info(f"Drop rate           |{best_hparams['drop_rate']}")
    logger.info(f"learning rate       |{best_hparams['learning_rate']}")
    logger.info("")
    # *****************************************
    # ハイパーパラメータの保存
    # *****************************************
    config["hyper_parameters"] = {
        "model": {
            "n_features": best_hparams["n_features"],
            "n_conv_hidden_layer": best_hparams["n_conv_hidden_layer"],
            "n_dense_hidden_layer": best_hparams["n_dense_hidden_layer"],
            "graph_dim": best_hparams["graph_dim"],
            "dense_dim": best_hparams["dense_dim"],
            "drop_rate": best_hparams["drop_rate"],
            "gnn_type": best_hparams["gnn_type"],
            "num_of_outputs": best_hparams["num_of_outputs"],
        },
        "other": {
            "learning_rate": best_hparams["learning_rate"]
        }
    }
    with open(model_dir.joinpath("all_params.yml"), mode="w") as f:
        yaml.dump(config, f)
    # *****************************************
    # モデルの本学習
    # *****************************************
    model = MolecularGNN(
        n_features=best_hparams["n_features"],
        n_conv_hidden_layer=best_hparams["n_conv_hidden_layer"],
        n_dense_hidden_layer=best_hparams["n_dense_hidden_layer"],
        graph_dim=best_hparams["graph_dim"],
        dense_dim=best_hparams["dense_dim"],
        gnn_type=best_hparams["gnn_type"],
        num_of_outputs=best_hparams["num_of_outputs"],
    )
    lr = best_hparams["learning_rate"]

    logger.info("***Training of the best model.***")
    training_datasets = graph_dataset.split_dataset("holdout")

    trained_model = training_model(
        model,
        output_dir,
        training_datasets,
        lr,
        n_epochs,
        batch_size,
        early_stopping_patience,
        device
    )
    logger.info("Training Finishued!!!")
    # *****************************************
    # モデルの評価
    # *****************************************
    evaluate_model(
        trained_model,
        logger,
        output_dir,
        training_datasets,
        batch_size,
        device,
    )
    copy_model_params(output_dir)
