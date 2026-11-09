"""V2: a two tower neural model with category features for unseen items."""

import numpy as np
import torch
from torch import nn
from .common import ModelData


class TwoTowerNet(nn.Module):
    def __init__(self, n_items: int, n_categories: int, dim: int):
        super().__init__()
        self.item = nn.Embedding(n_items, dim)
        self.category = nn.Embedding(n_categories, dim)
        self.user_tower = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_tower = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def item_vectors(self, item_ids, categories, known):
        return self.item_tower(self.item(item_ids) * known[..., None] + self.category(categories))

    def forward(self, history, history_mask, targets, categories, known):
        vectors = self.item_vectors(history, categories[history], known[history])
        context = (vectors * history_mask[:, :, None]).sum(1) / history_mask.sum(1, keepdim=True).clamp(min=1)
        users = self.user_tower(context)
        candidates = self.item_vectors(targets, categories[targets], known[targets])
        return (users * candidates).sum(-1) / users.shape[-1] ** 0.5


class NeuralTwoTower:
    name = "v2_neural"

    def __init__(self, dim=32, epochs=3, max_history=20, seed=42, max_pairs=300000):
        self.dim, self.epochs, self.max_history, self.seed, self.max_pairs = dim, epochs, max_history, seed, max_pairs

    def fit(self, data: ModelData):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() and len(data.items) > 1000 else "cpu")
        self.net = TwoTowerNet(len(data.items), len(data.category_names), self.dim).to(self.device)
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
            for batch_ids in np.array_split(rng.permutation(len(examples)), max(1, int(np.ceil(len(examples) / 256)))):
                batch = [examples[i] for i in batch_ids]
                length = max(len(h) for h, _ in batch)
                histories = torch.zeros((len(batch) * 2, length), dtype=torch.long, device=self.device)
                mask = torch.zeros_like(histories, dtype=torch.float32)
                targets = []
                for j, (history, target) in enumerate(batch):
                    for row in (j, j + len(batch)):
                        histories[row, -len(history):] = torch.tensor(history, device=self.device)
                        mask[row, -len(history):] = 1
                    targets.append(target)
                negatives = rng.integers(0, len(data.items), len(batch))
                targets = torch.tensor(targets + negatives.tolist(), device=self.device)
                labels = torch.cat([torch.ones(len(batch)), torch.zeros(len(batch))]).to(self.device)
                logits = self.net(histories, mask, targets, self.categories, self.known)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.net.eval()
        with torch.no_grad():
            self.candidates = torch.cat([self.net.item_vectors(ids, self.categories[ids], self.known[ids])
                for ids in torch.arange(len(data.items), device=self.device).split(8192)])
        return self

    def score(self, user: str):
        data = self.data
        history = data.histories.get(user, [])[-self.max_history:]
        if not history:
            return data.popularity.copy()
        with torch.no_grad():
            seq = torch.tensor([history], dtype=torch.long, device=self.device)
            mask = torch.ones_like(seq, dtype=torch.float32)
            context = self.net.item_vectors(seq, self.categories[seq], self.known[seq]).mean(1)
            users = self.net.user_tower(context)[0]
            scores = (self.candidates @ users).cpu().numpy() / self.dim ** 0.5
        return scores + 0.05 * data.popularity
