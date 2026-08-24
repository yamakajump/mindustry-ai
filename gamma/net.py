"""The policy network behind Beta.

A convolutional trunk over the local window, concatenated with the global scalars, feeding
one head per action component plus a value head. This is the factored action space from
`docs/decisions/0001-full-action-space.md` made concrete.

Every head is masked before sampling. Masking is not a nicety here: Gym-uRTS shows it is
the difference between learning and not learning at all in a grid RTS, and the same is
true of a policy that spends its first million steps discovering that most tiles refuse
most blocks.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

#: Very negative rather than -inf: softmax over an all-masked head would produce NaN,
#: and one NaN anywhere in a batch destroys the whole update.
MASK_FILL = -1e8


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Orthogonal initialisation, as used by every reference PPO implementation."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
    return layer


class MaskedCategorical:
    """A categorical distribution restricted to legal choices."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if mask is not None:
            # A row with nothing legal would otherwise be uniform over illegal options.
            empty = ~mask.any(dim=-1, keepdim=True)
            mask = mask | empty
            logits = torch.where(mask, logits, torch.full_like(logits, MASK_FILL))
        self.distribution = Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(value)

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()


class PolicyNet(nn.Module):
    """Shared trunk, one head per action component, one value head."""

    def __init__(
        self,
        channels: int,
        window: int,
        globals_size: int,
        n_types: int,
        n_blocks: int,
        n_rotations: int = 4,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.window = window
        self.n_types = n_types
        self.n_blocks = n_blocks
        self.n_rotations = n_rotations

        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(),
        )
        spatial_features = 64 * (window // 8) * (window // 8)

        self.globals = nn.Sequential(
            layer_init(nn.Linear(globals_size, 64)), nn.ReLU(),
        )
        self.body = nn.Sequential(
            layer_init(nn.Linear(spatial_features + 64, hidden)), nn.ReLU(),
        )

        # Small initial weights on the action heads keep the first policy close to
        # uniform, which stops early training from committing to one action type.
        self.head_type = layer_init(nn.Linear(hidden, n_types), std=0.01)
        self.head_block = layer_init(nn.Linear(hidden, n_blocks), std=0.01)
        self.head_position = layer_init(nn.Linear(hidden, window * window), std=0.01)
        self.head_rotation = layer_init(nn.Linear(hidden, n_rotations), std=0.01)
        # Small init here too: a value head starting on large outputs produces a huge
        # first loss, and the update that corrects it wipes out the policy with it.
        self.head_value = layer_init(nn.Linear(hidden, 1), std=0.1)

    def features(self, spatial: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        # uint8 to float here rather than in the environment: it keeps the replay buffer
        # four times smaller, which matters far more than the cast.
        spatial = spatial.float() / 255.0
        return self.body(torch.cat([self.trunk(spatial), self.globals(globals_)], dim=1))

    def value(self, spatial: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        return self.head_value(self.features(spatial, globals_)).squeeze(-1)

    def act(
        self,
        spatial: torch.Tensor,
        globals_: torch.Tensor,
        masks: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate a full action.

        Returns the action, the summed log probability, the summed entropy and the value.
        Components are summed because one environment action is all four choices together:
        the probability of the action is the product of the parts.
        """
        features = self.features(spatial, globals_)

        heads = [
            MaskedCategorical(self.head_type(features), masks.get("type")),
            MaskedCategorical(self.head_block(features), masks.get("block")),
            MaskedCategorical(self.head_position(features), masks.get("position")),
            MaskedCategorical(self.head_rotation(features), masks.get("rotation")),
        ]

        if action is None:
            picks = [head.sample() for head in heads]
        else:
            picks = [action[:, i] for i in range(4)]

        log_prob = sum(head.log_prob(pick) for head, pick in zip(heads, picks))
        entropy = sum(head.entropy() for head in heads)
        value = self.head_value(features).squeeze(-1)

        return torch.stack(picks, dim=1), log_prob, entropy, value

    def to_env_action(self, action: torch.Tensor) -> np.ndarray:
        """Turn (type, block, flat position, rotation) into the environment's layout."""
        action = action.cpu().numpy()
        flat = action[:, 2]
        return np.stack([
            action[:, 0],
            action[:, 1],
            flat % self.window,
            flat // self.window,
            action[:, 3],
        ], axis=1)
