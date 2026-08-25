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
        position_set_of_type: tuple[int, ...] | None = None,
        block_of_type: tuple[bool, ...] | None = None,
        rotation_of_type: tuple[bool, ...] | None = None,
    ) -> None:
        super().__init__()
        self.window = window
        self.n_types = n_types
        self.n_blocks = n_blocks
        self.n_rotations = n_rotations

        # What each action type actually cares about. Registered as buffers so they travel
        # with the network to whichever device it lives on, and so a checkpoint carries the
        # action layout it was trained under rather than trusting the caller to rebuild it
        # the same way. Defaults keep every type looking at everything, which is what the
        # network did before it was told the difference.
        self.register_buffer("position_set_of_type", torch.as_tensor(
            position_set_of_type if position_set_of_type is not None else (0,) * n_types,
            dtype=torch.long), persistent=False)
        self.register_buffer("block_of_type", torch.as_tensor(
            block_of_type if block_of_type is not None else (True,) * n_types,
            dtype=torch.bool), persistent=False)
        self.register_buffer("rotation_of_type", torch.as_tensor(
            rotation_of_type if rotation_of_type is not None else (True,) * n_types,
            dtype=torch.bool), persistent=False)

        self.trunk = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, 5, stride=2, padding=2)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
        )

        self.globals = nn.Sequential(
            layer_init(nn.Linear(globals_size, 64)), nn.ReLU(),
        )
        # Pooled rather than flattened. A flattened trunk ties every weight to an absolute
        # offset in the window, so the same feature at a different place is a different
        # input; pooling asks what is around instead of where it is, which is the right
        # question for choosing what to do.
        self.body = nn.Sequential(
            layer_init(nn.Linear(64 + 64, hidden)), nn.ReLU(),
        )

        # Small initial weights on the action heads keep the first policy close to
        # uniform, which stops early training from committing to one action type.
        self.head_type = layer_init(nn.Linear(hidden, n_types), std=0.01)
        self.head_block = layer_init(nn.Linear(hidden, n_blocks), std=0.01)
        self.head_rotation = layer_init(nn.Linear(hidden, n_rotations), std=0.01)
        # Small init here too: a value head starting on large outputs produces a huge
        # first loss, and the update that corrects it wipes out the policy with it.
        self.head_value = layer_init(nn.Linear(hidden, 1), std=0.1)

        # The position head, fully convolutional.
        #
        # It used to be a dense layer over a flattened trunk, which means the network
        # learns "the tile at index 1477 is worth building on" rather than "a tile that
        # looks like this is worth building on". On one map those are the same statement.
        # On a new one only the second transfers, and the whole point of training across
        # hundreds of generated worlds is that the second is what gets learned.
        #
        # A convolution is equivariant to translation by construction: shift the window
        # and the logits shift with it. This is the same shape as the spatial action head
        # in AlphaStar and in Gym-uRTS, and it is standard for a reason.
        self.context = layer_init(nn.Linear(hidden, 32))
        self.decoder = nn.Sequential(
            layer_init(nn.Conv2d(64 + 32, 64, 3, padding=1)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            layer_init(nn.Conv2d(64, 32, 3, padding=1)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            layer_init(nn.Conv2d(32, 32, 3, padding=1)), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            layer_init(nn.Conv2d(32, 16, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(16, 1, 1), std=0.01),
        )

    def trunk_map(self, spatial: torch.Tensor) -> torch.Tensor:
        """The convolutional feature map, before anything collapses it."""
        # uint8 to float here rather than in the environment: it keeps the replay buffer
        # four times smaller, which matters far more than the cast.
        return self.trunk(spatial.float() / 255.0)

    def features(self, spatial: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        return self.body_of(self.trunk_map(spatial), globals_)

    def body_of(self, spatial_map: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        pooled = spatial_map.mean(dim=(2, 3))
        return self.body(torch.cat([pooled, self.globals(globals_)], dim=1))

    def positions(self, spatial_map: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """One logit per tile of the window.

        The pooled context is broadcast back across the map before decoding, so the choice
        of tile can depend on what the agent is carrying and what it can afford, and not
        only on what the tile looks like.
        """
        context = self.context(features)[:, :, None, None]
        context = context.expand(-1, -1, spatial_map.shape[2], spatial_map.shape[3])
        logits = self.decoder(torch.cat([spatial_map, context], dim=1))
        return logits.flatten(1)

    def value(self, spatial: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        return self.head_value(self.features(spatial, globals_)).squeeze(-1)

    def _aimable(self, masks: dict[str, torch.Tensor],
                 kind: torch.Tensor) -> torch.Tensor | None:
        """The tiles the chosen action can actually target, one row per sample."""
        sets = masks.get("position_sets")
        if sets is None:
            # Callers that predate the named sets, and the tests that pin the old shape.
            return masks.get("position")

        which = self.position_set_of_type.to(kind.device)[kind]
        rows = torch.arange(sets.shape[0], device=sets.device)
        chosen = sets[rows, which]

        # A row with nothing legal would make the categorical undefined. It happens when
        # the type mask and the set disagree, which should not occur and must not crash a
        # run at three in the morning if it does.
        empty = ~chosen.any(dim=-1)
        if bool(empty.any()):
            chosen = chosen.clone()
            chosen[empty] = True
        return chosen

    @staticmethod
    def _relevant(mask: torch.Tensor | None, matters: torch.Tensor,
                  kind: torch.Tensor, width: int) -> torch.Tensor:
        """The real mask where the choice means something, one fixed option where it does not.

        A single allowed option makes the head deterministic: its log probability is zero
        and so is its entropy, so an irrelevant dimension stops adding noise to the ratio
        and stops being pushed towards uniform by the entropy bonus. Measured before this
        existed, the rotation head sat at 1.380 against a maximum of 1.386 across 1.5M
        steps and never moved, because nothing but the bonus ever touched it.
        """
        useful = matters.to(kind.device)[kind]
        if mask is None:
            mask = torch.ones((kind.shape[0], width), dtype=torch.bool, device=kind.device)

        pinned = torch.zeros_like(mask)
        pinned[:, 0] = True
        return torch.where(useful[:, None], mask, pinned)

    def act(
        self,
        spatial: torch.Tensor,
        globals_: torch.Tensor,
        masks: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
        per_head: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Sample or evaluate a full action.

        Returns the action, the summed log probability, the summed entropy and the value.
        Components are summed because one environment action is all four choices together:
        the probability of the action is the product of the parts.
        """
        spatial_map = self.trunk_map(spatial)
        features = self.body_of(spatial_map, globals_)

        # The type is decided first, and everything else is masked against it.
        #
        # Sampling all four independently means one position mask has to serve every type,
        # and there is no mask that can. `free` is right for building and exactly inverted
        # for breaking, so for as long as they shared one, every tile the agent could aim
        # at while breaking was guaranteed to hold nothing: 6,660 demolitions over 30
        # episodes, of which 23 hit a building it had placed, while `break` was 30% of
        # everything it did. The same independence made it pick a block and a rotation on
        # a `move`, which go nowhere and whose log probability lands in the ratio anyway.
        type_head = MaskedCategorical(self.head_type(features), masks.get("type"))
        kind = type_head.sample() if action is None else action[:, 0]

        heads = [
            type_head,
            MaskedCategorical(self.head_block(features),
                              self._relevant(masks.get("block"), self.block_of_type, kind,
                                             self.n_blocks)),
            MaskedCategorical(self.positions(spatial_map, features),
                              self._aimable(masks, kind)),
            MaskedCategorical(self.head_rotation(features),
                              self._relevant(masks.get("rotation"), self.rotation_of_type,
                                             kind, self.n_rotations)),
        ]

        if action is None:
            picks = [kind, *(head.sample() for head in heads[1:])]
        else:
            picks = [action[:, i] for i in range(4)]

        log_prob = sum(head.log_prob(pick) for head, pick in zip(heads, picks))
        entropy = sum(head.entropy() for head in heads)
        value = self.head_value(features).squeeze(-1)

        if per_head:
            # The summed entropy is what PPO optimises, and it is nearly useless as a
            # diagnostic: the position head ranges over the whole window, so its ln(2304)
            # swamps ln(6) for the type and ln(4) for the rotation. A policy that has
            # learnt exactly what to build and nothing about where to put it moves the
            # total by a few hundredths, which reads as a policy that has learnt nothing.
            spread = torch.stack([head.entropy().mean() for head in heads])
            return torch.stack(picks, dim=1), log_prob, entropy, value, spread

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
