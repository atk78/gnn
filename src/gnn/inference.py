from pathlib import Path

from rdkit import Chem
import numpy as np
import torch
import yaml

from gnn.mol2graph import Mol2Graph
from gnn.model import MolecularGNN




def inference(
    model_dir: str | Path,
    smiles_list: list,
    device: str = "cpu",
):
    if type(model_dir) is str:
        model_dir = Path(model_dir)
    params = yaml.safe_load(open(model_dir.joinpath("all_params.yaml")))
    polymer_flag = params["polymer"]
    if polymer_flag:
        smiles_array = [
            smiles if (smiles is not None) and (smiles.count("*") >= 2) else None
            for smiles in smiles_list
        ]
    else:
        smiles_array = [
            smiles if (smiles is not None) and (smiles.count("*") == 0) else None
            for smiles in smiles_list
        ]
    model = MolecularGNN(**params["hyper_parameters"]["model"])
    model.load_state_dict(torch.load(model_dir.joinpath("model_params.pth")))
    model = model.to(device)

    mol2graph = Mol2Graph(
        computable_atoms=params["computable_atoms"],
        poly_flag=params["polymer"],
        use_chirality=params["chirality"],
        use_stereochemistry=params["stereochemistry"],
    )

    n_node_features, n_edge_features = mol2graph.get_base_graph_features()
    smiles_array = np.array(
        [smiles for smiles in smiles_array if smiles is not None]
    )
    y_array = np.zeros(len(smiles_array))  # Dummy y values, not used in inference

    graph_dataset = mol2graph.get_graph_vectors(
        smiles_array=smiles_array,
        y_array=y_array,
        n_node_features=n_node_features,
        n_edge_features=n_edge_features,
    )

    model.eval()
    results = {smiles: None for smiles in smiles_list}
    with torch.no_grad():
        for graph, smiles in zip(graph_dataset, smiles_array):
            output = model(graph.to(device))
            results[smiles] = float(output.cpu().numpy()[0][0])
    return results
