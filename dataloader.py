import torch
from torch.utils.data import Dataset
import os
import pickle

def pad_tensor(tensor, target_shape):
    padded = torch.zeros(*target_shape, dtype=tensor.dtype)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(tensor.shape, target_shape))
    padded[slices] = tensor[slices]
    return padded



# Usage
class ScenegraphSequenceDataset(Dataset):
    def __init__(self, graph_dir, position_txt, sequence_length=5):
        self.graph_dir = graph_dir
        self.sequence_length = sequence_length

        self.graph_files = sorted([
            os.path.join(graph_dir, f)
            for f in os.listdir(graph_dir)
            if f.endswith(".pkl")
        ])

        with open(position_txt, "r") as f:
            self.positions = [
                torch.tensor([float(x), float(y), float(z)], dtype=torch.float)
                for _, x, y, z in (line.strip().split(",") for line in f)
            ]

        self.max_index = min(len(self.graph_files), len(self.positions)) - 1
        self.valid_range = self.max_index - self.sequence_length

    def __len__(self):
        return self.valid_range

    def __getitem__(self, idx):
        node_features_seq = []
        edge_index_seq = []
        batch_seq = []
        prev_positions_seq = []

        for i in range(idx, idx + self.sequence_length):
            with open(self.graph_files[i], "rb") as f:
                sg = pickle.load(f)

            node_feats = torch.tensor(sg["node_features"], dtype=torch.float)
            padded_node_feats = pad_tensor(node_feats, (9, 9))

            edge_index = torch.tensor(sg["edge_index"], dtype=torch.long)
            padded_edge_index = pad_tensor(edge_index, (2, 9))
            
            # padded_edge_index[padded_edge_index < 0] = 0


            # batch = torch.zeros(9, dtype=torch.long)  # fixed size

            node_features_seq.append(padded_node_feats)
            edge_index_seq.append(padded_edge_index)
            # batch_seq.append(batch)
            prev_positions_seq.append(self.positions[i])

        target_position = self.positions[idx + self.sequence_length]

        return {
            "node_features_seq": torch.stack(node_features_seq),        # [seq_len, 9, 9]
            "edge_index_seq": torch.stack(edge_index_seq),              # [seq_len, 2, 9]
            # "batch_seq": torch.stack(batch_seq),                        # [seq_len, 9]
            "prev_positions_seq": torch.stack(prev_positions_seq),      # [seq_len, 3]
            "target_position": target_position                          # [3]
        }
