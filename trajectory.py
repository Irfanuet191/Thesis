from dataloader import *
import sys, pdb
from pathlib import Path
# sys.path.append(str(Path("../../")))
import torch
import torch.nn as nn

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
# /home/irfan/roadscene2vec/roadscene2vec/data/dataset.py
from roadscene2vec.data.dataset import SceneGraphDataset
# from torch_geometric.data import Data, DataLoader, DataListLoader
from roadscene2vec.learning.util.metrics import get_metrics, log_wandb, log_wandb_transfer_learning 

scene_graph_dataset  = SceneGraphDataset()
# scene_graph_dataset.dataset_save_path ="/home/irfan/roadscene2vec/examples/object_based_sg_extraction_output.pkl"
# scene_graph_dataset_ = scene_graph_dataset.load() 
from torch.utils.data import DataLoader
from model import GCN_LSTM_PositionPredictor
from torch.utils.data import DataLoader, random_split

from torch.utils.data import DataLoader, random_split
graph_dir = "/home/irfan/roadscene2vec/examples/town2"
position_txt = "/home/irfan/roadscene2vec/examples/transferdata/pos.txt"

dataset = ScenegraphSequenceDataset(
    graph_dir=graph_dir,
    position_txt=position_txt,
    sequence_length=5
)
print(f"Dataset length: {len(dataset)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cuda"
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])



def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0]
    }

train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, collate_fn=collate_fn)

def rollout_with_dataset(model, dataset, start_idx=0, context_len=5, predict_horizon=10, device="cpu"):
    """
    Rollout trajectory predictions while fetching scene graphs from dataset.
    Returns predicted positions and ground truth positions.
    """
    model.eval()

    # Get the initial sequence (frames start_idx ... start_idx+context_len-1)
    sample = dataset[start_idx]
    node_features_seq = sample["node_features_seq"].unsqueeze(0).to(device)   # [1, T, N, F]
    edge_index_seq = sample["edge_index_seq"].unsqueeze(0).to(device)         # [1, T, 2, E]
    prev_positions_seq = sample["prev_positions_seq"].unsqueeze(0).to(device) # [1, T, D]

    predicted_positions = []
    ground_truth_positions = []

    with torch.no_grad():
        # Step 1: predict next after context
        pred = model(node_features_seq, edge_index_seq, prev_positions_seq)  # [1, D]
        predicted_positions.append(pred.squeeze(0).cpu())

        # ground truth for this step (frame after the context)
        gt = sample["target_position"]
        ground_truth_positions.append(gt.cpu())

    # Step 2: autoregressive rollout
    for step in range(1, predict_horizon):
        next_sample = dataset[start_idx + step]   # shifted sequence

        # get scene graph of the newest frame
        next_node_features = next_sample["node_features_seq"][-1:].unsqueeze(0).to(device)
        next_edge_index = next_sample["edge_index_seq"][-1:].unsqueeze(0).to(device)

        # build new sequences (drop oldest, append newest)
        prev_positions_seq = torch.cat(
            [prev_positions_seq[:, 1:], pred.unsqueeze(1)], dim=1
        )
        node_features_seq = torch.cat(
            [node_features_seq[:, 1:], next_node_features], dim=1
        )
        edge_index_seq = torch.cat(
            [edge_index_seq[:, 1:], next_edge_index], dim=1
        )

        # predict next
        with torch.no_grad():
            pred = model(node_features_seq, edge_index_seq, prev_positions_seq)
        predicted_positions.append(pred.squeeze(0).cpu())

        # collect ground truth for this frame
        ground_truth_positions.append(next_sample["target_position"].cpu())

    return (
        torch.stack(predicted_positions, dim=0),   # [predict_horizon, D]
        torch.stack(ground_truth_positions, dim=0) # [predict_horizon, D]
    )

model = GCN_LSTM_PositionPredictor()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model.to(device)

pred, gt = rollout_with_dataset(model, test_dataset, start_idx=0, predict_horizon=10, device=device)

print("Predicted positions shape:", pred.shape)   # [10, D]
print("Ground truth positions shape:", gt.shape) # [10, D]