"""V3: a compact HSTU-style sequence recommender.

It uses pointwise attention, relative position bias and gated pooling. It does not
include Meta's time bias, fused kernels, or production training stack.
"""

import numpy as np
import torch
from torch import nn
from .common import ModelData


class HSTUBlock(nn.Module):
    def __init__(self, dim, max_history):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, dim * 4)
        self.relative_position = nn.Embedding(max_history, 1)
        self.pool_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, x, mask):
        length = x.shape[1]
        q, k, v, gate = nn.functional.silu(self.project(self.input_norm(x))).chunk(4, dim=-1)
        positions = torch.arange(length, device=x.device)
        distance = (positions[:, None] - positions[None, :]).clamp(min=0)
        bias = self.relative_position(distance).squeeze(-1)
        attention = nn.functional.silu(q @ k.transpose(1, 2) / x.shape[-1] ** 0.5 + bias)
        allowed = (positions[:, None] >= positions[None, :])[None, :, :] & mask[:, None, :]
        pooled = (attention * allowed) @ v
        return x + self.output(self.pool_norm(pooled) * gate)


class SequenceNet(nn.Module):
    def __init__(self, n_items, n_categories, dim=32, max_history=20):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.category = nn.Embedding(n_categories, dim)
        self.encoder = HSTUBlock(dim, max_history)
        self.output = nn.Linear(dim, dim)

    def item_vectors(self, ids, categories, known):
        return self.item(ids + 1) * known[ids][..., None] + self.category(categories[ids])

    def context(self, seq, mask, categories, known):
        batch, length = seq.shape
        x = self.item_vectors(seq, categories, known)
        encoded = self.encoder(x, mask)
        last = mask.sum(1).long() - 1
        return self.output(encoded[torch.arange(batch), last])


class SequenceAttention:
    name = "v3_hstu_style"

    def __init__(self, dim=32, epochs=3, max_history=20, seed=42, max_pairs=300000):
        self.dim, self.epochs, self.max_history, self.seed, self.max_pairs = dim, epochs, max_history, seed, max_pairs

    def fit(self, data: ModelData):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() and len(data.items) > 1000 else "cpu")
        self.net = SequenceNet(len(data.items), len(data.category_names), self.dim, self.max_history).to(self.device)
        self.categories = torch.tensor(data.item_categories, dtype=torch.long, device=self.device)
        self.known = torch.tensor((data.popularity > 0).astype(float), dtype=torch.float32, device=self.device)
        examples = []
        for history in data.histories.values():
            for pos in range(1, len(history)):
                examples.append((history[max(0, pos - self.max_history):pos], history[pos]))
        if len(examples) > self.max_pairs:
            examples = [examples[i] for i in sorted(rng.choice(len(examples), self.max_pairs, replace=False))]
        if not examples:
            self.net.eval()
            with torch.no_grad():
                self.candidates = self.net.item_vectors(torch.arange(len(data.items), device=self.device), self.categories, self.known)
            return self
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.003)
        for _ in range(self.epochs):
            for batch_ids in np.array_split(rng.permutation(len(examples)), max(1, int(np.ceil(len(examples) / 128)))):
                batch = [examples[i] for i in batch_ids]
                length = max(len(h) for h, _ in batch)
                seq = torch.zeros((len(batch), length), dtype=torch.long, device=self.device)
                mask = torch.zeros((len(batch), length), dtype=torch.bool, device=self.device)
                for j, (history, _) in enumerate(batch):
                    seq[j, :len(history)] = torch.tensor(history, device=self.device)
                    mask[j, :len(history)] = True
                positives = torch.tensor([target for _, target in batch], device=self.device)
                negatives = torch.tensor(rng.integers(0, len(data.items), len(batch)), device=self.device)
                context = self.net.context(seq, mask, self.categories, self.known)
                pos = self.net.item_vectors(positives, self.categories, self.known)
                neg = self.net.item_vectors(negatives, self.categories, self.known)
                loss = nn.functional.softplus(-((context * (pos - neg)).sum(1) / self.dim ** 0.5)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.net.eval()
        with torch.no_grad():
            self.candidates = torch.cat([self.net.item_vectors(ids, self.categories, self.known)
                for ids in torch.arange(len(data.items), device=self.device).split(8192)])
        return self

    def score(self, user: str):
        data = self.data
        history = data.histories.get(user, [])[-self.max_history:]
        if not history:
            return data.popularity.copy()
        with torch.no_grad():
            seq = torch.tensor([history], dtype=torch.long, device=self.device)
            mask = torch.ones_like(seq, dtype=torch.bool)
            context = self.net.context(seq, mask, self.categories, self.known)[0]
            scores = (self.candidates @ context).cpu().numpy() / self.dim ** 0.5
        return scores + 0.05 * data.popularity
