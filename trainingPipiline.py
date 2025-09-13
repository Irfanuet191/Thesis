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

print(f"Train loader size: {len(train_loader)}, Test loader size: {len(test_loader)}")

model = GCN_LSTM_PositionPredictor()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
i=0
best_test_loss=40.0
for epoch in range(100):
    model.train()
    for batch in train_loader:
        # print(f"Processing batch {i}")
        i+=1
        # print(batch.keys())
        # print(f"Batch size: {batch['node_features_seq'].shape}, Time edge_index_seq: {batch['edge_index_seq'].shape}, Nodes prev_positions_seq: {batch['prev_positions_seq'].shape}, Features per node: {batch['target_position'].shape}")
        
        # for k in batch:
        #     batch[k] = batch[k]
            
        # print(f"Batch size: {batch['node_features_seq'].shape}, Time steps: {batch['node_features_seq'].shape}, Nodes per step: {batch['node_features_seq'].shape}, Features per node: {batch['node_features_seq'].shape}")
        # print(batch["batch_seq"])
        # try:
        pred=model(
                batch["node_features_seq"].to(device),
                batch["edge_index_seq"].to(device),
                # batch["batch_seq"].to(device),
                batch["prev_positions_seq"].to(device)
            )
        # except:
        #     print(f"skipped")
        # print("Forward pass completed.")
        # print(f"Batch size: {batch['node_features_seq'].shape}, Time steps: {batch['node_features_seq'].shape}, Nodes per step: {batch['node_features_seq'].shape}, Features per node: {batch['node_features_seq'].shape}")
        loss = loss_fn(pred, batch["target_position"].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    for batch in test_loader:
        # print(f"Processing batch {i}")
        i+=1
        # print(batch.keys())
        # print(f"Batch size: {batch['node_features_seq'].shape}, Time edge_index_seq: {batch['edge_index_seq'].shape}, Nodes prev_positions_seq: {batch['prev_positions_seq'].shape}, Features per node: {batch['target_position'].shape}")
        
        # for k in batch:
        #     batch[k] = batch[k]
            
        # print(f"Batch size: {batch['node_features_seq'].shape}, Time steps: {batch['node_features_seq'].shape}, Nodes per step: {batch['node_features_seq'].shape}, Features per node: {batch['node_features_seq'].shape}")
        # print(batch["batch_seq"])
        # try:
        pred=model(
                batch["node_features_seq"].to(device),
                batch["edge_index_seq"].to(device),
                # batch["batch_seq"].to(device),
                batch["prev_positions_seq"].to(device)
            )
        # except:
        #     print(f"skipped")
        # print("Forward pass completed.")
        # print(f"Batch size: {batch['node_features_seq'].shape}, Time steps: {batch['node_features_seq'].shape}, Nodes per step: {batch['node_features_seq'].shape}, Features per node: {batch['node_features_seq'].shape}")
        loss_test = loss_fn(pred, batch["target_position"].to(device))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
    

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Test Loss = {loss_test.item():.4f}")

    # ---- save best model ----
    if loss_test.item() < best_test_loss:
        best_test_loss = loss_test.item()
        torch.save(model.state_dict(), "best_model_weights.pth")
        print(f"✅ Saved new best model at epoch {epoch+1} with Test Loss {best_test_loss:.4f}")
model.load_state_dict(torch.load("model_weights.pth"))

