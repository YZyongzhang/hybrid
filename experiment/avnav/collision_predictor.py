"""Action-conditioned collision-risk predictor used as a learned AVNav feature."""

import torch
from torch import nn
from torch.nn import functional as F


class ActionCollisionPredictor(nn.Module):
    """Predict collision probability for (state, previous action, candidate action)."""

    def __init__(self, embedding_dim=256, action_count=4, hidden_dim=256):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.action_count = int(action_count)
        input_dim = self.embedding_dim + 2 * self.action_count
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embedding, previous_action, candidate_action):
        """Return collision logits for explicit candidate actions."""
        features = torch.cat((embedding, previous_action, candidate_action), dim=-1)
        return self.network(features).squeeze(-1)

    def all_action_probabilities(self, embedding, previous_action, batch_size=4096):
        """Return [batch, action_count] probabilities in AVNav action-ID order."""
        count = len(embedding)
        result = []
        action_ids = torch.arange(self.action_count, device=embedding.device)
        candidates = F.one_hot(action_ids, self.action_count).to(embedding.dtype)
        self.eval()
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                stop = min(count, start + batch_size)
                size = stop - start
                expanded_embedding = embedding[start:stop, None, :].expand(
                    size, self.action_count, self.embedding_dim
                ).reshape(-1, self.embedding_dim)
                expanded_previous = previous_action[start:stop, None, :].expand(
                    size, self.action_count, self.action_count
                ).reshape(-1, self.action_count)
                expanded_candidates = candidates[None, :, :].expand(
                    size, self.action_count, self.action_count
                ).reshape(-1, self.action_count)
                logits = self(expanded_embedding, expanded_previous,
                              expanded_candidates)
                result.append(torch.sigmoid(logits).reshape(size, self.action_count))
        return torch.cat(result, dim=0) if result else embedding.new_empty((0, self.action_count))


class MultiHorizonActionCollisionPredictor(ActionCollisionPredictor):
    """Predict immediate and short-horizon collision risk for each candidate."""

    def __init__(self, embedding_dim=256, action_count=4, hidden_dim=256):
        super().__init__(embedding_dim, action_count, hidden_dim)
        input_dim = self.embedding_dim + 2 * self.action_count
        self.future_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_future(self, embedding, previous_action, candidate_action):
        features = torch.cat((embedding, previous_action, candidate_action), dim=-1)
        return self.future_network(features).squeeze(-1)

    def all_future_action_probabilities(self, embedding, previous_action, batch_size=4096):
        count = len(embedding)
        result = []
        action_ids = torch.arange(self.action_count, device=embedding.device)
        candidates = F.one_hot(action_ids, self.action_count).to(embedding.dtype)
        self.eval()
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                stop = min(count, start + batch_size)
                size = stop - start
                expanded_embedding = embedding[start:stop, None, :].expand(
                    size, self.action_count, self.embedding_dim
                ).reshape(-1, self.embedding_dim)
                expanded_previous = previous_action[start:stop, None, :].expand(
                    size, self.action_count, self.action_count
                ).reshape(-1, self.action_count)
                expanded_candidates = candidates[None, :, :].expand(
                    size, self.action_count, self.action_count
                ).reshape(-1, self.action_count)
                logits = self.forward_future(expanded_embedding, expanded_previous,
                                             expanded_candidates)
                result.append(torch.sigmoid(logits).reshape(size, self.action_count))
        return torch.cat(result, dim=0) if result else embedding.new_empty((0, self.action_count))
