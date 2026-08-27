import pytest
from pydantic import ValidationError

from upscaler.schemas import GENERATIVE_MODES, JobSettings, ProcessingMode


def test_sharpen_only_requires_a_strength() -> None:
    with pytest.raises(ValidationError, match="requires a non-zero sharpen strength"):
        JobSettings(processing_mode=ProcessingMode.sharpen_only, sharpen=0)


def test_sharpen_only_rejects_neural_only_settings() -> None:
    with pytest.raises(ValidationError, match="does not accept neural"):
        JobSettings(processing_mode=ProcessingMode.sharpen_only, sharpen=20, tta=True)

    with pytest.raises(ValidationError, match="does not accept neural"):
        JobSettings(processing_mode=ProcessingMode.sharpen_only, sharpen=20, tile_size=256)


def test_the_default_job_is_faithful() -> None:
    """Nothing generative may happen to a user who never touched a control."""
    settings = JobSettings()
    assert settings.processing_mode == ProcessingMode.upscale
    assert settings.processing_mode not in GENERATIVE_MODES
    assert settings.target_edge == 3840


def test_no_mode_in_this_repository_claims_to_generate() -> None:
    """This build only reconstructs, and the label exists to keep that checkable.

    A mode added here that invented detail would have to join GENERATIVE_MODES
    to be labelled honestly, and this test is where that decision surfaces.
    """
    assert frozenset() == GENERATIVE_MODES
    for mode in ProcessingMode:
        assert mode not in GENERATIVE_MODES


def test_unknown_settings_are_rejected_rather_than_ignored() -> None:
    """A stale client must fail loudly instead of silently getting a default."""
    with pytest.raises(ValidationError):
        JobSettings(engine="spandrel-sr")


@pytest.mark.parametrize("passes", [0, 5])
def test_pass_budget_is_bounded(passes: int) -> None:
    with pytest.raises(ValidationError):
        JobSettings(max_neural_passes=passes)


def test_tile_size_is_zero_or_workable() -> None:
    with pytest.raises(ValidationError, match="at least 32"):
        JobSettings(tile_size=16)
    assert JobSettings(tile_size=0).tile_size == 0


def test_a_workflow_belongs_only_to_the_mode_that_runs_one() -> None:
    """It drives an external graph, which is the one thing Upscale must never do."""
    for mode in (ProcessingMode.upscale, ProcessingMode.sharpen_only):
        with pytest.raises(ValidationError, match="only applies to illustration"):
            JobSettings(processing_mode=mode, workflow="illustration-upscale", sharpen=15)


def test_illustration_accepts_a_workflow() -> None:
    settings = JobSettings(
        processing_mode=ProcessingMode.illustration,
        workflow="illustration-upscale",
    )
    assert settings.workflow == "illustration-upscale"
