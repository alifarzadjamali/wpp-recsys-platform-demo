"""V3: a compact causal attention recommender inspired by HSTU's sequence setup.

This is a demo scale implementation, not Meta's production HSTU kernel or model.
"""

import numpy as np
import torch
from torch import nn
from .common import ModelData


class SequenceNet(nn.Module):
    def __init__(self, n_items, n_categories, dim=32, max_history=20):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.category = nn.Embedding(n_categories, dim)
        self.position = nn.Embedding(max_history, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2, batch_first=True, dropout=0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.output = nn.Linear(dim, dim)

    def item_vectors(self, ids, categories, known):
        return self.item(ids + 1) * known[ids][..., None] + self.category(categories[ids])

    def context(self, seq, mask, categories, known):
        batch, length = seq.shape
        x = self.item_vectors(seq, categories, known) + self.position(torch.arange(length))[None, :, :]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(x, mask=causal, src_key_padding_mask=~mask)
        last = mask.sum(1).long() - 1
        return self.output(encoded[torch.arange(batch), last])


class SequenceAttention:
    name = "v3_sequence_attention"

    def __init__(self, dim=32, epochs=3, max_history=20, seed=42):
        self.dim, self.epochs, self.max_history, self.seed = dim, epochs, max_history, seed

    def fit(self, data: ModelData):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.data = data
        self.net = SequenceNet(len(data.items), len(data.category_names), self.dim, self.max_history)
        self.categories = torch.tensor(data.item_categories, dtype=torch.long)
        self.known = torch.tensor((data.popularity > 0).astype(float), dtype=torch.float32)
        examples = []
        for history in data.histories.values():
            for pos in range(1, len(history)):
                examples.append((history[max(0, pos - self.max_history):pos], history[pos]))
        if not examples:
            return self
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.003)
        for _ in range(self.epochs):
            for batch_ids in np.array_split(rng.permutation(len(examples)), max(1, int(np.ceil(len(examples) / 128)))):
                batch = [examples[i] for i in batch_ids]
                length = max(len(h) for h, _ in batch)
                seq = torch.zeros((len(batch), length), dtype=torch.long)
                mask = torch.zeros((len(batch), length), dtype=torch.bool)
                for j, (history, _) in enumerate(batch):
                    seq[j, :len(history)] = torch.tensor(history)
                    mask[j, :len(history)] = True
                positives = torch.tensor([target for _, target in batch])
                negatives = torch.tensor(rng.integers(0, len(data.items), len(batch)))
                context = self.net.context(seq, mask, self.categories, self.known)
                pos = self.net.item_vectors(positives, self.categories, self.known)
                neg = self.net.item_vectors(negatives, self.categories, self.known)
                loss = nn.functional.softplus(-((context * (pos - neg)).sum(1) / self.dim ** 0.5)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def score(self, user: str):
        data = self.data
        history = data.histories.get(user, [])[-self.max_history:]
        if not history:
            return data.popularity.copy()
        with torch.no_grad():
            seq = torch.tensor([history], dtype=torch.long)
            mask = torch.ones_like(seq, dtype=torch.bool)
            context = self.net.context(seq, mask, self.categories, self.known)[0]
            ids = torch.arange(len(data.items))
            candidates = self.net.item_vectors(ids, self.categories, self.known)
            scores = (candidates @ context).numpy() / self.dim ** 0.5
        return scores + 0.05 * data.popularity
