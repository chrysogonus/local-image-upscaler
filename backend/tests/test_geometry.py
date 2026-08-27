import math

import pytest

from upscaler.geometry import (
    MAX_NATIVE_SCALE,
    NATIVE_SCALES,
    RESIDUAL_TOLERANCE,
    choose_native_scale,
    estimate_working_bytes,
    plan_native_scales,
    target_dimensions,
)


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ((1920, 1080), 3840, (3840, 2160)),
        ((1080, 1920), 3840, (2160, 3840)),
        ((4032, 3024), 7680, (7680, 5760)),
        ((8000, 1), 3840, (3840, 1)),
    ],
)
def test_target_dimensions_preserve_aspect(source, target, expected):
    width, height, _ = target_dimensions(*source, target)
    assert (width, height) == expected


def test_target_dimensions_reject_invalid_values():
    with pytest.raises(ValueError):
        target_dimensions(0, 20, 3840)


@pytest.mark.parametrize(("requested", "native"), [(1.1, 2), (2, 2), (2.1, 3), (3.1, 4), (20, 4)])
def test_native_scale_is_bounded(requested, native):
    assert choose_native_scale(requested) == native


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (8, (4, 2)),  # 480x270 to 4K: no 2x Lanczos tail left over
        (16, (4, 4)),
        (19.2, (4, 4, 2)),
        (4, (4,)),
        (2, (2,)),
        (2.5, (3,)),
    ],
)
def test_plan_covers_the_requested_factor_with_native_passes(requested, expected):
    plan = plan_native_scales(requested)
    assert plan == expected
    assert math.prod(plan) >= requested


def test_plan_matches_the_single_pass_choice_within_one_pass():
    """Nothing at or below 4x should change behaviour; only the tail is new."""
    for requested in (1.1, 1.5, 2, 2.1, 3, 3.5, 4):
        assert plan_native_scales(requested) == (choose_native_scale(requested),)


def test_plan_stops_once_the_remainder_is_negligible():
    # 4.1x is a 4x pass and a 1.025x resize; a second pass would cost a full
    # inference for a factor the final resize handles invisibly.
    assert plan_native_scales(4.1) == (4,)
    assert plan_native_scales(RESIDUAL_TOLERANCE) == (2,)


@pytest.mark.parametrize("max_passes", [1, 2, 3, 4])
def test_plan_never_exceeds_the_allowed_pass_count(max_passes):
    plan = plan_native_scales(60, max_passes)
    assert 1 <= len(plan) <= max_passes
    assert set(plan) <= set(NATIVE_SCALES)
    assert math.prod(plan) <= MAX_NATIVE_SCALE**max_passes


def test_plan_is_empty_when_no_enlargement_is_needed():
    assert plan_native_scales(1) == ()
    assert plan_native_scales(0.4) == ()


def test_plan_rejects_a_zero_pass_budget():
    with pytest.raises(ValueError):
        plan_native_scales(4, 0)


def test_plan_rejects_an_engine_with_no_native_scale():
    with pytest.raises(ValueError):
        plan_native_scales(4, 3, ())


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(2, (4,)), (4, (4,)), (8, (4, 4)), (19.2, (4, 4, 4))],
)
def test_a_four_times_only_engine_never_gets_asked_for_another_scale(requested, expected):
    """The NCNN runtime returns a wrongly cropped image for -s 2 and -s 3.

    It reports the right dimensions while doing it, so the failure is invisible
    unless the plan is constrained to what the engine actually produces.
    """
    plan = plan_native_scales(requested, 3, (4,))
    assert plan == expected
    assert set(plan) == {4}
    assert math.prod(plan) >= requested


def test_chained_estimate_exceeds_the_single_pass_estimate():
    single = estimate_working_bytes(200, 200, 3840, 3840, neural=True, tile_size=128, passes=(4,))
    chained = estimate_working_bytes(
        200, 200, 3840, 3840, neural=True, tile_size=128, passes=(4, 4, 2)
    )
    assert chained > single


def test_neural_memory_estimate_accounts_for_tile_working_set():
    classical = estimate_working_bytes(100, 100, 400, 400, neural=False, tile_size=0)
    neural = estimate_working_bytes(100, 100, 400, 400, neural=True, tile_size=128)
    assert neural > classical
