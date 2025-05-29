import itertools
from typing import Any

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, Atom, Bond
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import rdmolops
import numpy as np
from numpy.typing import NDArray
import torch
from torch_geometric.data import Data


class Mol2Graph:
    def __init__(
        self,
        computable_atoms: list[str],
        poly_flag: bool = False,
        use_chirality: bool = False,
        use_stereochemistry: bool = False
    ):
        self.computable_atoms = computable_atoms
        self.poly_flag = poly_flag,
        self.use_chirality = use_chirality
        self.use_stereochemstry = use_stereochemistry

    def get_graph_vectors(
        self,
        smiles_array: NDArray,
        y_array: NDArray,
        n_node_features: int,
        n_edge_features: int
    ):
        """
        SMILES文字列の配列を指定されたノードおよびエッジ特徴量の数を使用して
        グラフベクトルに変換する。

        Parameters
        ----------
        smiles_array : NDArray
            分子構造を表すSMILES文字列の配列。
        y_array : NDArray
            各SMILES文字列に対応するターゲット値の配列。
        n_node_features : int
            グラフ内の各ノードに使用する特徴量の数。
        n_edge_features : int
            グラフ内の各エッジに使用する特徴量の数。

        Returns
        -------
        list
            各分子のグラフ構造と特徴量を表すグラフベクトルのリスト。
        """
        graph_vectors = list()
        for i_smiles, y in zip(smiles_array, y_array):
            i_graph_vector = self.calc_graph_vector(
                i_smiles, y, n_node_features, n_edge_features
            )
            graph_vectors.append(i_graph_vector)
        return graph_vectors

    def get_base_graph_features(self) -> tuple[int, int]:
        """
        ベースとなるグラフの特徴量を取得するメソッド。
        このメソッドは、分子グラフのノード特徴量とエッジ特徴量の次元数を計算します。
        ポリマー構造を考慮する場合は、ポリマー結合のインデックスを検索し、
        原子が主鎖か側鎖かを設定した後に計算を行う。

        Returns
        -------
        int
            ノード特徴量の次元数。
        int
            エッジ特徴量の次元数。
        """
        # 初期設定
        if self.poly_flag:
            test_mol = Chem.MolFromSmiles("*CC(*)c1ccccc1")
            poly_bond_idx = self.search_poly_bond_idx(test_mol)
            test_mol = self.set_chain_type(test_mol, poly_bond_idx)
        else:
            test_mol = Chem.MolFromSmiles("c1ccccc1")
        ComputeGasteigerCharges(test_mol)
        # 数を確定させるための前処理
        n_node_features = len(
            self.get_atom_features(test_mol.GetAtomWithIdx(0))
        )
        n_edge_features = len(
            self.get_bond_features(test_mol.GetBondBetweenAtoms(0, 1))
        )
        return n_node_features, n_edge_features

    def calc_graph_vector(
        self,
        smiles: str,
        y: float,
        n_node_features: int,
        n_edge_features: int,
    ):
        """グラフベクトルを計算する

        Parameters
        ----------
        smiles : str
            グラフ化したい分子のSMILES文字列
        y : float
            グラフ化したい分子の物性
        n_node_features : int
            ノードの特徴量の数
        n_edge_features : int
            エッジの特徴量の数

        Returns
        -------
        Data
            GraphVectorオブジェクト
        """
        mol = Chem.MolFromSmiles(smiles)
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        if self.poly_flag:
            poly_bond_idx = self.search_poly_bond_idx(mol)
            mol = self.set_chain_type(mol, poly_bond_idx)
        ComputeGasteigerCharges(mol, nIter=100, throwOnParamFailure=False)

        # ****************** グラフの特徴量の計算 ******************
        # ノードの特徴(X)
        X = np.zeros((n_nodes, n_node_features))
        for idx in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(idx)
            X[idx, :] = self.get_atom_features(atom)
        X = torch.from_numpy(X).to(dtype=torch.float32)

        # エッジのindex(E)
        rows, cols = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int32))
        torch_cols = torch.from_numpy(cols.astype(np.int32))
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # エッジの特徴(EF)
        EF = np.zeros((n_edges, n_edge_features))
        for k, (idx1, idx2) in enumerate(zip(rows, cols)):
            EF[k] = self.get_bond_features(
                bond=mol.GetBondBetweenAtoms(int(idx1), int(idx2)),
            )
        EF = torch.from_numpy(EF).to(dtype=torch.float32)

        # 物性
        Y = torch.tensor(np.array([y]), dtype=torch.float32)

        # グラフベクトルの作成
        graph_vector = Data(x=X, edge_index=E, edge_attr=EF, y=Y)
        return graph_vector

    def one_hot_encoding(self, x: Any, encode_list: list[Any]):
        """encode_listの各要素に応じてxをエンコードする

        Parameters
        ----------
        x : Any
            エンコードしたい値
        encode_list : list[Any]
            エンコードしたい値のリスト

        Returns
        -------
        list[int]
            xをエンコードしたリスト
        """
        if x not in encode_list:
            x = encode_list[-1]
        # encode_listの各要素に対してatomが適応しているかどうか(True:1, False:0)を判別し
        # それらをエンコードとして返す
        binary_encoding = [
            int(boolian_value)
            for boolian_value in list(map(lambda s: x == s, encode_list))
        ]
        return binary_encoding

    def search_poly_bond_idx(self, mol: Mol) -> list[int]:
        """ポリマーの結合を表す[*]のインデックスを取得する

        Parameters
        ----------
        mol : Mol
            ポリマー結合部を探索するmolオブジェクト

        Returns
        -------
        list[int]
            ポリマーの結合[*]のインデックスのリスト（基本2つ）
        """
        poly_bond_mol = Chem.MolFromSmiles("*")  # 「*」のmolファイルを生成
        # 結合[*]のindexを取得 -> 例: ((1,), (2,))
        poly_bond_idx = mol.GetSubstructMatches(poly_bond_mol)
        # 結合[*]のindexを平坦化する ((1,), (2,)) -> [1, 2]
        poly_bond_idx = list(
            itertools.chain.from_iterable(poly_bond_idx)
        )
        return poly_bond_idx

    def replace_poly_bond_to_atoms(
        self, mol: Mol, poly_bond_idx: list[int]
    ):
        # 結合を隣の原子に置換
        adgacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
        next_atoms = []
        for idx in poly_bond_idx:
            # ポリマー結合[*]の隣の原子を取得
            next_atom_idx = int(np.where(adgacency_matrix[idx] == 1)[0][0])
            next_atom = mol.GetAtomWithIdx(next_atom_idx).GetSymbol()
            if next_atom == "Si":
                next_atom = "[SiH3]"
            next_atoms.append(next_atom)
        next_atoms.reverse()
        # Replace Substructs
        for next_atom in next_atoms:
            replace_from = Chem.MolFromSmiles("*")
            replace_to = Chem.MolFromSmiles(next_atom)
            mol = AllChem.ReplaceSubstructs(mol, replace_from, replace_to)[0]
        mol = Chem.RemoveHs(mol)  # 余計なHを削除
        return mol

    def replace_poly_bond_to_h(
        self, mol: Mol, poly_bond_index_list: list[int]
    ):
        for _ in poly_bond_index_list:
            replace_from = Chem.MolFromSmiles("*")
            replace_to = Chem.MolFromSmiles("[H]")
            mol = AllChem.ReplaceSubstructs(mol, replace_from, replace_to)[0]
        mol = Chem.RemoveHs(mol)  # たまにHが余計についたりするため
        return mol

    def set_chain_type(self, mol: Mol, poly_bond_idx: list[int]):
        """重合の結合[*]ともう一方の重合の結合[*]の最短パスを取得し、その間に存在する原子群を主鎖とし、
        その他を側鎖としてmolにプロパティを追加する。

        Parameters
        ----------
        mol : Mol
            設定したいmolオブジェクト
        poly_bond_idx : list[int]
            重合の結合[*]のインデックスのリスト（基本2つ）

        Returns
        -------
        Mol
            主鎖と側鎖のプロパティが設定されたmolオブジェクト
        """
        # 重合の結合[*]ともう一方の重合の結合[*]の最短パスを取得する
        poly_bond_shortest_path = list(
            itertools.permutations(poly_bond_idx, 2)
        )
        main_chain_idx: set[int] = set()
        # 最短パスから主鎖構造のindexを取得する
        for poly_bond_idx_i in poly_bond_shortest_path:
            main_chain_idx = main_chain_idx | set(rdmolops.GetShortestPath(
                mol, poly_bond_idx_i[0], poly_bond_idx_i[1]
            ))
        # 環の中に主鎖の原子が含まれている場合、主鎖に追加する
        sssr = Chem.GetSymmSSSR(mol)  # 環情報を取得
        for i in range(len(sssr)):
            if not main_chain_idx.isdisjoint(set(sssr[i])):
                main_chain_idx = main_chain_idx | set(sssr[i])
        function_search_smarts_mol = Chem.MolFromSmarts("[*]=[*]")
        search_idx = mol.GetSubstructMatches(function_search_smarts_mol)
        for idx in search_idx:
            if not main_chain_idx.isdisjoint(set(idx)):
                main_chain_idx = main_chain_idx | set(idx)
        # 原子にプロパティを追加
        for atom in mol.GetAtoms():
            if atom.GetIdx() in main_chain_idx:
                atom.SetProp("chain_type", "main_chain")
            else:
                atom.SetProp("chain_type", "sub_chain")
        return mol

    # def calc_bond_ff_features(self, mol: Mol, idx1: int, idx2: int):
    #     # rdkitに搭載しているUFF ForceFieldのパラメータを取得
    #     bond_kb, bond_r0 = rdForceFieldHelpers.GetUFFBondStretchParams(
    #         mol, int(idx1), int(idx2)
    #     )
    #     bond_vdw_x, bond_vdw_d = rdForceFieldHelpers.GetUFFVdWParams(
    #         mol, int(idx1), int(idx2)
    #     )
    #     bond_kb = (bond_kb - 940.610377) / 204.924599
    #     bond_r0 = (bond_r0 - 1.414844) / 0.089204
    #     bond_vdw_x = (bond_vdw_x - 3.811359) / 0.079849
    #     bond_vdw_d = (bond_vdw_d - 0.100566) / 0.017500
    #     bond_ff_features = [bond_kb, bond_r0, bond_vdw_x, bond_vdw_d]
    #     return bond_ff_features

    def get_atom_features(self, atom: Atom):
        """原子（ノード）の特徴量を取得する

        Parameters
        ----------
        atom : Atom
            原子の情報を持つrdkitのAtomオブジェクト

        Returns
        -------
        np.ndarray
            ノードの特徴量ベクトル
        """
        # if hydrogens_implicit == False:
        #     computable_atoms = ["H"] + computable_atoms

        # 原子タイプのエンコーディング C -> [0, 0, 1, 0, 0 ...]のように変更
        atom_type_enc = self.one_hot_encoding(
            str(atom.GetSymbol()), self.computable_atoms
        )
        # 水素の数
        n_hydrogens_enc = self.one_hot_encoding(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3]
        )
        # 原子に隣接する原子の数
        n_heavy_neighbors_enc = self.one_hot_encoding(
            int(atom.GetDegree()), [1, 2, 3, 4]
        )
        # 電荷
        formal_carge_enc = self.one_hot_encoding(
            int(atom.GetFormalCharge()), [-1, 0, 1]
        )
        # 混成軌道の種類
        hybridization_type_enc = self.one_hot_encoding(
            atom.GetHybridization(),
            [
                rdkit.Chem.rdchem.HybridizationType.S,
                rdkit.Chem.rdchem.HybridizationType.SP,
                rdkit.Chem.rdchem.HybridizationType.SP2,
                rdkit.Chem.rdchem.HybridizationType.SP3,
                rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
            ],
        )
        # 原子の価数
        n_valence_enc = self.one_hot_encoding(
            atom.GetImplicitValence(), [0, 1, 2, 3]
        )
        # 環状の原子かどうか
        is_in_a_ring_enc = [int(atom.IsInRing())]

        # 環の大きさ
        if atom.IsInRing():
            ring_size = 7
            for i in range(3, 7):
                if atom.IsInRingSize(i):
                    ring_size = i
            if ring_size == 3:
                ring_size_enc = [0, 1, 0, 0, 0, 0]
            elif ring_size == 4:
                ring_size_enc = [0, 0, 1, 0, 0, 0]
            elif ring_size == 5:
                ring_size_enc = [0, 0, 0, 1, 0, 0]
            elif ring_size == 6:
                ring_size_enc = [0, 0, 0, 0, 1, 0]
            else:
                ring_size_enc = [0, 0, 0, 0, 0, 1]
        else:
            ring_size_enc = [1, 0, 0, 0, 0, 0]

        # 芳香族かどうか
        # is_aromatic_enc = [int(atom.GetIsAromatic())]
        if atom.GetIsAromatic():
            neighbor_aromatic_num = [
                na.GetIsAromatic() for na in atom.GetNeighbors()
            ].count(True)
            # 通常の芳香族
            if neighbor_aromatic_num == 2:
                is_aromatic_enc = [0, 1, 0]
            # ナフタレンのような芳香族の場合
            else:
                is_aromatic_enc = [0, 0, 1]
        # 芳香族ではない
        else:
            is_aromatic_enc = [1, 0, 0]
        # 原子の重さ
        atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]
        # 原子のvdw体積
        vdw_radius = [
            float(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()))
        ]
        # 共有結合半径
        covalent_radius = [
            float(Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()))
        ]
        # 原子の電荷
        if self.poly_flag:
            atom_charge = []
        else:
            atom_charge = [
                float(atom.GetProp("_GasteigerCharge"))
            ]
        if self.poly_flag:
            if atom.GetProp("chain_type") == "main_chain":
                main_chain_atom_enc = [0]  # 主鎖の場合0
            else:
                main_chain_atom_enc = [1]
        else:
            main_chain_atom_enc = []

        # 原子の特徴量ベクトル
        atom_feature_vector = (
            atom_type_enc  # 原子の種類(encoding)
            + n_heavy_neighbors_enc  # 隣接する原子の数(encoding)
            + n_hydrogens_enc  # 水素の数(encoding)
            + formal_carge_enc  # 電荷(encoding)
            + hybridization_type_enc  # 混成軌道の種類(encoding)
            + n_valence_enc  # 原子の価数(encoding)
            + is_in_a_ring_enc  # 環状の原子かどうか(encoding)
            + ring_size_enc  # 環の大きさ(encoding)
            + is_aromatic_enc  # 芳香族かどうか(encoding)
            + atomic_mass_scaled  # 原子の重さ
            + vdw_radius  # 原子のvdw体積
            + covalent_radius  # 共有結合半径
            + atom_charge  # 原子の電荷
            + main_chain_atom_enc  # 主鎖の原子かどうか(encoding)
        )
        # chiralityの選択
        if self.use_chirality:
            chirality_type_enc = self.one_hot_encoding(
                atom.GetChiralTag(),
                [
                    "CHI_UNSPECIFIED",
                    "CHI_TETRAHEDRAL_CW",
                    "CHI_TETRAHEDRAL_CCW",
                    "CHI_OTHER",
                ],
            )
            atom_feature_vector + chirality_type_enc

        return np.array(atom_feature_vector)

    def get_bond_features(self, bond: Bond):
        """エッジの特徴量を取得する

        Parameters
        ----------
        bond : Bond
            エッジの情報を持つrdkitのBondオブジェクト

        Returns
        -------
        np.ndarray
            エッジの特徴量ベクトル
        """
        bond_type_enc = self.one_hot_encoding(
            bond.GetBondType(),
            [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ]
        )
        # 共役結合かどうか
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        # 結合が環状かどうか
        bond_is_in_ring_enc = [int(bond.IsInRing())]

        # 結合の特徴量ベクトル
        bond_features_vector = (
            bond_type_enc
            + bond_is_conj_enc
            + bond_is_in_ring_enc
            # + ff_features
        )

        if self.use_stereochemstry:
            stereo_type_enc = self.one_hot_encoding(
                bond.GetStereo(),
                [
                    rdkit.Chem.rdchem.BondStereo.STEREONONE,
                    rdkit.Chem.rdchem.BondStereo.STEREOZ,
                    rdkit.Chem.rdchem.BondStereo.STEREOE,
                ],
            )
            bond_features_vector += stereo_type_enc
        return np.array(bond_features_vector)
