# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool

# class GCN_LSTM_PositionPredictor(nn.Module):
#     def __init__(self, in_dim, gcn_hidden_dim=32, lstm_hidden_dim=64, mlp_hidden_dim=64):
#         super().__init__()

#         self.gcn1 = GCNConv(in_dim, gcn_hidden_dim)
#         self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

#         self.lstm = nn.LSTM(input_size=gcn_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)

#         self.mlp = nn.Sequential(
#             nn.Linear(lstm_hidden_dim + 3, mlp_hidden_dim),
#             nn.ReLU(),
#             nn.Linear(mlp_hidden_dim, 3)
#         )

#     def forward(self, node_features_seq, edge_index_seq, prev_positions_seq):
#         """
#         node_features_seq:   (B, T, N, F)
#         edge_index_seq:      (B, T, 2, E)
#         batch_seq:           (B, T, N)
#         prev_positions_seq:  (B, T, 3)

#         Returns:
#             output: (B, 3)
#         """
#         # B, T, N, FE = node_features_seq.size()
#         # print(f"Batch size: {B}, Time steps: {T}, Nodes per step: {N}, Features per node: {FE}")
#         # outputs = []

#         # for b in range(B):
#         #     graph_embeds = []

#             # for t in range(T):
#                 # print(node_features_seq.shape, edge_index_seq.shape)
#                 # print(f"Processing batch {b}, time step {t}")
#         x = F.relu(self.gcn1(node_features_seq, edge_index_seq))
#         # x = F.relu(self.gcn2(x, edge_index_seq))
#         # pooled = x.mean(dim=0, keepdim=True)  # simple mean over all nodes → [1, D]
#                 # graph_embeds.append(pooled)
#         return x
            
#             # graph_embeds = torch.stack(graph_embeds, dim=0)  # [T, D]
#             # # graph_embeds = graph_embeds.unsqueeze(0)         # [1, T, D]  
#             # # print(f"Graph embeds shape: {graph_embeds.shape}")        
#             # lstm_out, _ = self.lstm(graph_embeds)            # [1, T, H]
#             # # print(f"LSTM output shape: {lstm_out.shape}")
#             # lstm_last = lstm_out[0, -1]   # [H]
#             # # print(f"LSTM last output shape: {lstm_last.shape}")
#             # # print(f"Previous positions shape: {prev_positions_seq[b, -1].shape}")

#             # combined = torch.cat([lstm_last, prev_positions_seq[b, -1]], dim=-1)  # [H + 3]
#             # out = self.mlp(combined.unsqueeze(0))  # [1, 3]
#             # outputs.append(out.squeeze(0))         # [3]

#         return torch.stack(outputs, dim=0)  # [B, 3]


import torch
import torch.nn as nn
import torch.nn.functional as Functional
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
def clean_edges(edge_index):
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    mask = (edge_index >= 0).all(dim=0)
    return edge_index[:, mask]

class GCN_LSTM_PositionPredictor(nn.Module):
    def __init__(self, in_dim=9, gcn_hidden_dim=32, lstm_hidden_dim=64, mlp_hidden_dim=64):
        super().__init__()

        # --- Graph Convolutions ---
        self.gcn1 = GCNConv(in_dim, gcn_hidden_dim)
        self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim)

        # --- Temporal Model (LSTM) ---
        self.lstm = nn.LSTM(input_size=gcn_hidden_dim + 3,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True)

        # --- Final Prediction Head ---
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 3)
        )

    def forward(self, node_feats_seq, edge_index_seq, prev_pos_seq):
        """
        node_feats_seq : [B, T, N, F]
        edge_index_seq : [B, T, 2, E]
        prev_pos_seq   : [B, T, 3]
        """
        B, T, N, F = node_feats_seq.shape
        graph_embeds = []

        for t in range(T):
            # Collect all graphs at time t into a batch
            data_list = []
            # for b in range(B):
            #     x = node_feats_seq[b, t]                     # [N, F]
            #     edge_index = edge_index_seq[b, t].long()     # [2, E]
            #     data_list.append(Data(x=x, edge_index=edge_index))
            # data_list = []
            # for b in range(B):
            #     x = node_feats_seq[b, t]                     # [N, F]
            #     edge_index = edge_index_seq[b, t].long()

            #     # ✅ Ensure edges are valid
            #     if edge_index.max() >= x.size(0):
            #         print(f"[WARN] Invalid edge index: max={edge_index.max()} >= num_nodes={x.size(0)}")
            #         mask = edge_index < x.size(0)
            #         edge_index = edge_index[:, mask.all(dim=0)]  # remove bad edges

            #     data_list.append(Data(x=x, edge_index=edge_index))
            for b in range(B):
                x = node_feats_seq[b, t]                   # [N, F]
                edge_index = edge_index_seq[b, t].long()

                # Fix 1-based indexing
                if edge_index.max() == x.size(0):
                    edge_index -= 1

                # Validate edges
                mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
                edge_index = edge_index[:, mask]
                edge_index = clean_edges(edge_index)

                data_list.append(Data(x=x, edge_index=edge_index))
            
            

            # Create a batch of graphs
            batch = Batch.from_data_list(data_list)  # merges graphs
            # print(f'batch size: {batch.x.shape}, edge_index: {batch.edge_index}')
            x = Functional.relu(self.gcn1(batch.x, batch.edge_index))
            x = self.gcn2(x, batch.edge_index)

            # Mean pooling per graph
            pooled = torch.zeros(B, x.size(-1), device=x.device)
            pooled = scatter_mean(x, batch.batch, dim=0, out=pooled)

            graph_embeds.append(pooled)  # [B, hidden_dim]

        graph_embeds = torch.stack(graph_embeds, dim=1)  # [B, T, hidden_dim]

        # Combine with previous positions
        lstm_input = torch.cat([graph_embeds, prev_pos_seq], dim=-1)  # [B, T, hidden_dim+3]

        # Temporal modeling
        lstm_out, _ = self.lstm(lstm_input)
        last_out = lstm_out[:, -1, :]  # [B, lstm_hidden]

        # Predict future position
        pred = self.mlp(last_out)  # [B, 3]
        return pred
# B, T, N, F1, E = 20, 5, 9, 9, 9
# model = GCN_LSTM_PositionPredictor()

# node_feats_seq = torch.randn(B, T, N, F1)
# edge_index_seq = torch.randint(0, N, (B, T, 2, E))
# prev_pos_seq = torch.randn(B, T, 3)

# pred = model(node_feats_seq, edge_index_seq, prev_pos_seq)
# print("Predicted future positions:", pred.shape)  # [20, 3]

# data_list = []
# for b in range(B):
#     x = node_feats_seq[b, t]                     # [N, F]
#     edge_index = edge_index_seq[b, t].long()

#     # ✅ Ensure edges are valid
#     if edge_index.max() >= x.size(0):
#         print(f"[WARN] Invalid edge index: max={edge_index.max()} >= num_nodes={x.size(0)}")
#         mask = edge_index < x.size(0)
#         edge_index = edge_index[:, mask.all(dim=0)]  # remove bad edges

#     data_list.append(Data(x=x, edge_index=edge_index))

# batch = Batch.from_data_list(data_list)

