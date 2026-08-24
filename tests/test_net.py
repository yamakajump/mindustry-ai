"""The policy network, and the one property it has to keep.

Choosing a tile is the action that decides whether an agent has learned a map or the game.
A dense head over a flattened trunk learns "index 1477 is a good place to build", which is
true on one map and meaningless on the next. A convolutional head learns "a tile that looks
like this is a good place to build", which is the thing that transfers.
"""

from __future__ import annotations

import torch

from gamma.net import PolicyNet


def net(window: int = 48) -> PolicyNet:
    return PolicyNet(channels=14, window=window, globals_size=12, n_types=6, n_blocks=6)


def masks(batch: int, window: int) -> dict[str, torch.Tensor]:
    return {
        "type": torch.ones(batch, 6, dtype=torch.bool),
        "block": torch.ones(batch, 6, dtype=torch.bool),
        "position": torch.ones(batch, window * window, dtype=torch.bool),
    }


def test_an_action_carries_all_four_choices() -> None:
    policy = net()
    spatial = torch.randint(0, 255, (4, 14, 48, 48), dtype=torch.uint8)
    action, log_prob, entropy, value = policy.act(spatial, torch.randn(4, 12), masks(4, 48))

    assert action.shape == (4, 4)
    assert log_prob.shape == (4,) and entropy.shape == (4,) and value.shape == (4,)


def test_the_position_head_does_not_scale_with_the_window() -> None:
    """The regression this guards against is the whole point of the head.

    A dense position head has one weight per tile per hidden unit, so doubling the window
    quadruples the network. A convolutional one has the same weights whatever the window,
    because it asks the same question at every tile. If this test starts failing, the head
    has gone back to memorising offsets.
    """
    small = sum(p.numel() for p in net(32).parameters())
    large = sum(p.numel() for p in net(64).parameters())

    assert small == large, f"{small} against {large}: the head is tied to the window size"


def test_the_position_logits_cover_the_window() -> None:
    policy = net()
    spatial = torch.randint(0, 255, (2, 14, 48, 48), dtype=torch.uint8)
    feature_map = policy.trunk_map(spatial)
    logits = policy.positions(feature_map, policy.body_of(feature_map, torch.randn(2, 12)))

    assert logits.shape == (2, 48 * 48)


def test_a_fully_masked_head_does_not_produce_nan() -> None:
    """A head with nothing legal happens: an agent with an empty core can build nowhere.
    Softmax over an all-masked row would be NaN, and one NaN destroys the whole update."""
    policy = net()
    blocked = masks(2, 48)
    blocked["position"] = torch.zeros(2, 48 * 48, dtype=torch.bool)

    _, log_prob, entropy, _ = policy.act(
        torch.randint(0, 255, (2, 14, 48, 48), dtype=torch.uint8), torch.randn(2, 12), blocked
    )

    assert torch.isfinite(log_prob).all() and torch.isfinite(entropy).all()


def test_a_flat_position_becomes_a_tile() -> None:
    policy = net()
    action = torch.tensor([[1, 2, 48 * 3 + 7, 1]])

    assert policy.to_env_action(action)[0].tolist() == [1, 2, 7, 3, 1]
