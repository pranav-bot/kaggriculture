from kaggriculture.actions.controller import ActionController
from kaggriculture.helpers.opponent import clone_like
from kaggriculture.helpers.market_overlays import reset_market_overlay_state


def test_clone_like_detects_similar_farms():
    tiles = [[{"kind": "PLANT", "crop": "WHEAT"}] + [None] * 9 for _ in range(10)]
    obs = {
        "player": 0,
        "farms": [
            {"hands": [], "unlocked_quadrants": ["NW"], "tiles": tiles},
            {"hands": [], "unlocked_quadrants": ["NW"], "tiles": tiles},
        ],
    }
    assert clone_like(obs) is True


def test_postprocess_runs_without_microstructure_collision():
    reset_market_overlay_state()
    controller = ActionController(enable_market_microstructure=False)
    orders = controller._postprocess_market(
        {"player": 0, "farms": [{}, {}], "market": {"prices": {"WHEAT": 25}}},
        [["SELL", "MELON", 2], ["SELL", "WHEAT", 2]],
        [["PASS"]],
        ["PASS"],
        [],
        [],
    )
    assert len(orders) == 2
