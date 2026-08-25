"""The policy network, and the one property it has to keep.

Choosing a tile is the action that decides whether an agent has learned a map or the game.
A dense head over a flattened trunk learns "index 1477 is a good place to build", which is
true on one map and meaningless on the next. A convolutional head learns "a tile that looks
like this is a good place to build", which is the thing that transfers.
"""

from __future__ import annotations

import math

import torch

from gamma.net import PolicyNet


def positions_of(net: PolicyNet, window: int, channels: int = 3,
                 globals_size: int = 2) -> int:
    """How many tiles the position head actually offers.

    Asked of the network rather than assumed to be window squared: the trunk has a stride,
    so a small window does not map one logit to one tile and a test that guesses fails on
    a shape mismatch that says nothing about what it was checking.
    """
    spatial = torch.zeros(1, channels, window, window)
    spatial_map = net.trunk_map(spatial)
    globals_ = torch.zeros(1, globals_size)
    return net.positions(spatial_map, net.body_of(spatial_map, globals_)).shape[1]


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


def test_the_position_mask_follows_the_action_type() -> None:
    """Sampling four choices independently means one mask must serve every type, and none can.

    `free` is right for building and exactly inverted for breaking, because a tile that is
    buildable and empty is by definition a tile with nothing on it to break. For as long
    as they shared one mask, every tile the agent could aim at while breaking held nothing:
    6,660 demolitions over 30 archived episodes, 23 of which hit a building it had placed,
    while `break` was 30% of everything it did.
    """
    window = 16
    net = PolicyNet(channels=3, window=window, globals_size=2, n_types=3, n_blocks=2,
                    position_set_of_type=(0, 1, 0),
                    block_of_type=(True, False, False),
                    rotation_of_type=(True, False, False))
    tiles = positions_of(net, window)

    # Two sets that share no tile, so the mask that was used can be read off the choice.
    build_here = torch.zeros(1, 2, tiles, dtype=torch.bool)
    build_here[0, 0, :4] = True
    build_here[0, 1, 8:12] = True

    for kind, allowed in ((0, range(0, 4)), (1, range(8, 12))):
        masks = {
            "type": torch.zeros(1, 3, dtype=torch.bool),
            "position_sets": build_here,
        }
        masks["type"][0, kind] = True

        picks, _, _, _ = net.act(torch.zeros(1, 3, window, window),
                                 torch.zeros(1, 2), masks)
        assert int(picks[0, 0]) == kind
        assert int(picks[0, 2]) in allowed, (
            f"type {kind} aimed outside the set it is supposed to target"
        )


def test_an_irrelevant_choice_costs_nothing() -> None:
    """A `move` still picks a block and a rotation, and they go nowhere.

    Their log probability lands in the importance ratio regardless, which is noise, and the
    entropy bonus keeps pushing them towards uniform because no gradient ever tells them
    otherwise: the rotation head sat at 1.380 against a maximum of 1.386 across 1.5M steps
    and never moved. Pinned to one option they contribute exactly zero to both.
    """
    window = 16
    net = PolicyNet(channels=3, window=window, globals_size=2, n_types=2, n_blocks=5,
                    position_set_of_type=(0, 0),
                    block_of_type=(True, False),
                    rotation_of_type=(True, False))
    tiles = positions_of(net, window)

    masks = {
        "type": torch.tensor([[False, True]]),        # only the irrelevant type is legal
        "position_sets": torch.ones(1, 1, tiles, dtype=torch.bool),
        "block": torch.ones(1, 5, dtype=torch.bool),
    }
    picks, log_prob, entropy, _ = net.act(torch.zeros(1, 3, window, window),
                                          torch.zeros(1, 2), masks)

    assert int(picks[0, 1]) == 0 and int(picks[0, 3]) == 0, "pinned to the one option"
    # Only the position head can carry entropy here: type is forced and the other two are
    # pinned, so anything above ln(16) would mean an irrelevant head is still spending.
    assert float(entropy[0].detach()) <= math.log(tiles) + 1e-5
