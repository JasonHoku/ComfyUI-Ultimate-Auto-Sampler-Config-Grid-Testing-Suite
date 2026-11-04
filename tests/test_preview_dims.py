import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_utils import preview_dims


def test_half_scale_portrait():
    assert preview_dims(832, 1216, 0.5, 8) == (416, 608)
    assert preview_dims(832, 1216, 0.5, 16) == (416, 608)


def test_third_scale_snaps_to_ratio16():
    assert preview_dims(1000, 1000, 0.33, 16) == (336, 336)


def test_dims_are_multiples_of_ratio_and_never_below_one_cell():
    for scale in (0.25, 0.33, 0.5, 0.75):
        for ratio in (8, 16):
            w, h = preview_dims(64, 96, scale, ratio)
            assert w % ratio == 0 and h % ratio == 0
            assert w >= ratio and h >= ratio
    assert preview_dims(8, 8, 0.05, 16) == (16, 16)


def test_scale_one_is_identity_and_bad_ratio_means_8():
    assert preview_dims(1001, 777, 1.0, 16) == (1001, 777)
    assert preview_dims(832, 1216, 0.5, 0) == (416, 608)
    assert preview_dims(832, 1216, 0.5, None) == (416, 608)
