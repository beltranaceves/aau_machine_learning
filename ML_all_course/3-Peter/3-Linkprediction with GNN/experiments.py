import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
import random
from torch_geometric.datasets import Planetoid


def activation_by_name(name):
    """Return a PyTorch activation module given a short name.

    This helper lets experiment code pick activation functions by string
    (e.g., 'relu', 'elu', 'gelu'). If an unknown name is supplied,
    the function returns ReLU as default.
    """
    if name is None or name.lower() == 'relu':
        return nn.ReLU()
    if name.lower() == 'elu':
        return nn.ELU()
    if name.lower() == 'gelu':
        return nn.GELU()
    return nn.ReLU()


class ConfigurableGCNConv(MessagePassing):
    """GCN-style message passing layer with configurable options.

    Parameters
    - in_channels, out_channels: sizes for the linear transform
    - aggr: aggregation operator passed to MessagePassing ('add','mean','max')
    - nonlin: activation name used after aggregation
    - bias: whether to add a learnable bias after aggregation
    - use_self_loops: whether to add self-loops before message passing
    - use_norm: whether to apply symmetric degree normalization like standard GCN

    Notes on normalization:
    - When `use_norm` is True we compute deg^{-1/2}_i deg^{-1/2}_j and pass
      that per-edge scaling to `propagate` via the `norm` argument. The
      `message()` method multiplies incoming messages by that factor.
    - When `use_norm` is False we explicitly pass `norm=None` to `propagate`
      so the `message()` signature is always the same (this avoids runtime
      signature errors in PyG which expects the same set of args).
    """

    def __init__(self, in_channels, out_channels, aggr='add', nonlin='relu', bias=False, use_self_loops=True, use_norm=False):
        super().__init__(aggr=aggr)
        # Linear transform applied before message passing (as in GCN)
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.nonlin = activation_by_name(nonlin)
        self.use_self_loops = use_self_loops
        self.use_norm = use_norm
        if bias:
            # small learnable bias applied after aggregation
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index):
        # Optionally add self-loops so nodes aggregate their own transformed features
        if self.use_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linear transform (W * x)
        x = self.lin(x)

        # Compute symmetric normalization factors if requested and pass to propagate
        if self.use_norm:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            out = self.propagate(edge_index, x=x, norm=norm)
        else:
            # pass `norm=None` explicitly so message() always receives the kwarg
            out = self.propagate(edge_index, x=x, norm=None)

        # nonlinearity and optional bias
        out = self.nonlin(out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, norm=None):
        """Message function for PyG's MessagePassing.

        x_j contains the source node features for each edge. If `norm` is
        provided (per-edge scalar) we scale messages by that factor; otherwise
        we return raw messages. This implements the normalization in GCN.
        """
        if norm is None:
            return x_j
        else:
            return norm.view(-1, 1) * x_j


class ConfigurableGNN(nn.Module):
    """Simple GNN wrapper that stacks configurable message-passing layers.

    The network returns a pairwise similarity matrix produced by taking the
    dot-product of final node embeddings. This similarity matrix is used as
    the link-prediction score between nodes.
    """

    def __init__(self, dims, aggr='add', nonlin='relu', bias=False, use_self_loops=True, use_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        # Create `depth` convolutional layers followed by a final Linear
        for i in range(len(dims) - 2):
            self.layers.append(ConfigurableGCNConv(dims[i], dims[i+1], aggr=aggr, nonlin=nonlin, bias=bias, use_self_loops=use_self_loops, use_norm=use_norm))
        # Final linear maps to the desired embedding size
        self.layers.append(Linear(dims[-2], dims[-1], bias=bias))

    def forward(self, x, edge_index):
        h = x
        for l in self.layers:
            if isinstance(l, ConfigurableGCNConv):
                h = l(h, edge_index)
            else:
                h = l(h)
        # Return similarity scores (node_i x node_j)
        sim = torch.matmul(h, h.t())
        return sim


def build_adj_from_edge_index(edge_index, num_nodes):
    """Build a dense adjacency matrix (torch.Tensor) from an edge_index.

    This is used as the training target for the simple MSE loss we use in
    the notebook experiments. For large graphs you would not use a dense
    target, but for Karate Club (N=34) this is convenient and simple.
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    rows = edge_index[0].cpu().numpy().astype(int)
    cols = edge_index[1].cpu().numpy().astype(int)
    for r, c in zip(rows, cols):
        A[r, c] = 1.0
    return A


def hit_rate_at_k(sim_matrix, hidden_edges, k=10):
    """Compute Hit Rate@K for a set of hidden (u,v) test edges.

    For each hidden edge (u,v) we rank candidate target nodes for source u by
    descending similarity and count it as a hit if v appears among the top-k.
    The final metric is the fraction of hidden edges that are hits.
    """
    sim = sim_matrix.detach().cpu().numpy()
    hits = 0
    for u, v in hidden_edges:
        scores = sim[u]
        rank = np.argsort(scores)[::-1]  # descending
        # find position of v
        pos = int(np.where(rank == v)[0][0])
        if pos < k:
            hits += 1
    return hits / max(1, len(hidden_edges))


def train_model(model, features, train_edge_index, target_adj, epochs=200, lr=1e-3, weight_decay=0):
    """Train the model to fit the adjacency matrix using MSE loss.

    This simple training loop is intentionally minimal for didactic reasons:
    - Uses full-batch gradient descent (suitable for small graphs)
    - Optimizes MSE between predicted similarity matrix and binary adjacency
      target (a proxy objective for link prediction experiments)
    - Returns the final predicted similarity matrix (model in eval mode)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss(reduction='mean')
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        out = model(features, train_edge_index)
        l = loss_fn(out, target_adj)
        l.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(features, train_edge_index)
    return out


def run_karate_search(features, train_edge_index, hidden_edges, epochs=200, k=10,
                      layer_depths=[2,3], hidden_dims=[4,8], aggrs=['add','mean'],
                      activations=['relu'], biases=[False], self_loops=[True], norms=[False], lr=1e-3):
    """Run a simple grid search over GNN hyperparameters on the Karate Club graph.

    - `features`: [N, F] torch tensor of node features
    - `train_edge_index`: edge_index for the training graph (hidden edges removed)
    - `hidden_edges`: list of (u,v) pairs that were removed for testing

    The function returns a list of result dicts sorted by Hit Rate@K (descending).
    Each dict contains the hyperparameters and the observed hit rate.
    """
    results = []
    N = features.size(0)
    target_adj = build_adj_from_edge_index(train_edge_index, N)  # adjacency of training graph

    # Iterate over the provided hyperparameter grid and train a fresh model
    # for each configuration. Keep the implementation simple and explicit so it
    # is easy to explain during an exam.
    for depth in layer_depths:
        for hdim in hidden_dims:
            # dims defines the layer sizes: [in, hid, ..., out]
            dims = [features.size(1)] + [hdim] * (depth - 1) + [hdim]
            for aggr in aggrs:
                for act in activations:
                    for bias in biases:
                        for sl in self_loops:
                            for norm in norms:
                                model = ConfigurableGNN(dims=dims, aggr=aggr, nonlin=act, bias=bias, use_self_loops=sl, use_norm=norm)
                                out = train_model(model, features, train_edge_index, target_adj, epochs=epochs, lr=lr)
                                hr = hit_rate_at_k(out, hidden_edges, k=k)
                                results.append({
                                    'depth': depth,
                                    'hidden_dim': hdim,
                                    'aggr': aggr,
                                    'activation': act,
                                    'bias': bias,
                                    'self_loops': sl,
                                    'norm': norm,
                                    'hit_rate': hr
                                })
    results = sorted(results, key=lambda x: x['hit_rate'], reverse=True)
    return results


def _split_undirected_edge_index(edge_index, remove_frac=0.2, seed=None):
    """Given a PyG edge_index, return a new edge_index with a fraction of
    undirected edges removed and the list of removed (u,v) pairs.

    We treat edges as undirected pairs (min,max) to avoid double-removing
    both directions separately.
    """
    if seed is not None:
        random.seed(seed)
    rows = edge_index[0].cpu().numpy().astype(int)
    cols = edge_index[1].cpu().numpy().astype(int)
    seen = set()
    undirected = []
    for r, c in zip(rows, cols):
        a, b = (r, c) if r <= c else (c, r)
        if (a, b) not in seen:
            seen.add((a, b))
            undirected.append((a, b))

    num_remove = max(1, int(len(undirected) * remove_frac))
    hidden = random.sample(undirected, num_remove)
    hidden_set = set(hidden)

    keep_idx = []
    for idx, (r, c) in enumerate(zip(rows, cols)):
        a, b = (r, c) if r <= c else (c, r)
        if (a, b) in hidden_set:
            continue
        keep_idx.append(idx)

    keep_edge_index = edge_index[:, keep_idx]
    return keep_edge_index, hidden


def apply_configs_on_dataset(features, edge_index, hidden_edges, configs, epochs=100, k=10, lr=1e-3):
    """Apply a list of hyperparameter configs on a given dataset (features, edge_index).

    `configs` is a list of dicts with keys: depth, hidden_dim, aggr, activation, bias, self_loops, norm
    Returns sorted results with hit rates.
    """
    results = []
    N = features.size(0)
    target_adj = build_adj_from_edge_index(edge_index, N)
    for cfg in configs:
        depth = cfg['depth']
        hdim = cfg['hidden_dim']
        dims = [features.size(1)] + [hdim] * (depth - 1) + [hdim]
        model = ConfigurableGNN(dims=dims, aggr=cfg['aggr'], nonlin=cfg['activation'], bias=cfg['bias'], use_self_loops=cfg['self_loops'], use_norm=cfg['norm'])
        out = train_model(model, features, edge_index, target_adj, epochs=epochs, lr=lr)
        hr = hit_rate_at_k(out, hidden_edges, k=k)
        res = cfg.copy()
        res.update({'hit_rate': hr})
        results.append(res)
    return sorted(results, key=lambda x: x['hit_rate'], reverse=True)


def run_cora_experiment(configs, root='/home/toita86/Documents/AAU/ML/data/Planetoid', remove_frac=0.2, epochs=100, k=10, lr=1e-3, seed=None):
    """Run Phase 2 experiments on the Cora dataset using provided configs.

    - `configs`: list of hyperparameter dicts (same format as Karate results)
    - `root`: path to Planetoid dataset root (uses local data if present)
    - `remove_frac`: fraction of undirected edges to hold-out for testing

    Returns sorted results for the provided configs on Cora.
    """
    dataset = Planetoid(root, 'Cora')
    data = dataset[0]
    features = data.x
    edge_index = data.edge_index

    # Split edges (undirected) into train and hidden test set
    train_edge_index, hidden_edges = _split_undirected_edge_index(edge_index, remove_frac=remove_frac, seed=seed)

    # Apply given configs on the Cora train graph
    results = apply_configs_on_dataset(features, train_edge_index, hidden_edges, configs, epochs=epochs, k=k, lr=lr)
    return results
