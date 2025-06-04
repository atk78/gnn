# GNN

## 概要

本パッケージはグラフニューラルネットワーク（GNN: Graph Neural Network）を用いて、分子のSMILES表記から各種物性値を予測します。分子構造をグラフとして表現し、GNNで分子の特徴を抽出することで、従来の手法よりも高精度な予測を実現します。SMILES（Simplified Molecular Input Line Entry System）は分子構造をテキストで表現するための標準的な記法であり、本パッケージはSMILESを入力として物性値の推定を可能にします。

## 環境設定

本プロジェクトはDockerを用いて環境構築を行います。.devcontainerディレクトリを備えており、Visual Studio Codeの拡張機能「Dev Containers」を利用することで、簡単に開発環境を立ち上げることができます。

1. [Docker](https://www.docker.com/)をインストールしてください。
2. [Visual Studio Code](https://code.visualstudio.com/)と拡張機能「Dev Containers」をインストールしてください。
3. プロジェクトフォルダをVSCodeで開き、「Reopen in Container」または「Open Folder in Container」を実行してください。

## 使いかた

本パッケージには以下のJupyter Notebookが含まれています。
`notebook/`ディレクトリ内に配置されています。

- **preprocessing.ipynb**
  データの前処理を行います。SMILESデータのクリーニングや特徴量生成などを行います。

- **train.ipynb**
  前処理済みデータを用いてGNNモデルの学習を行います。

- **predict.ipynb**
  学習済みモデルを使って新たなSMILESデータの物性値を予測します。

### 実行例

