import socket
from pathlib import Path

import pytest

from upscaler import maintenance
from upscaler.maintenance import (
    COMFYUI_KEEP,
    CleanupRefused,
    Target,
    collect,
    collect_comfyui,
    collect_workspaces,
    comfyui_root,
    is_comfyui_install,
    remove,
    remove_paths,
    skipped_symlinks,
)


def _comfyui(root: Path) -> Path:
    """A directory that passes the install check, with the dirs a wipe touches."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("# ComfyUI\n")
    (root / "comfy").mkdir()
    for name in ("input", "output", "temp"):
        (root / name).mkdir()
    (root / "user" / "default" / "workflows").mkdir(parents=True)
    return root


def _job_workspace(root: Path, name: str) -> Path:
    workdir = root / name
    workdir.mkdir(parents=True)
    (workdir / "source.upload").write_bytes(b"x" * 100)
    (workdir / "result.png").write_bytes(b"y" * 200)
    return workdir


def test_every_job_directory_is_collected_and_the_root_survives(tmp_path):
    work_root = tmp_path / "work"
    work_root.mkdir()
    _job_workspace(work_root, "aaaaaaaa-0000")
    _job_workspace(work_root, "bbbbbbbb-1111")

    target = collect_workspaces(work_root)
    assert target is not None
    assert target.files == 4
    assert target.size == 600

    remove_paths(target.paths)
    assert work_root.is_dir()  # the root is reused, only its children go
    assert list(work_root.iterdir()) == []


def test_an_absent_work_root_collects_nothing(tmp_path):
    assert collect_workspaces(tmp_path / "never-created") is None


def test_comfyui_directories_are_emptied_but_kept(tmp_path):
    root = _comfyui(tmp_path / "ComfyUI")
    (root / "input" / "photo.png").write_bytes(b"a" * 50)
    (root / "temp" / "ComfyUI_temp_abcde_00001_.png").write_bytes(b"b" * 70)
    (root / "user" / "default" / "workflows" / "saved_graph.json").write_text("{}")

    targets = collect_comfyui(root)
    remove_paths([path for target in targets for path in target.paths])

    for name in ("input", "output", "temp"):
        assert (root / name).is_dir()
        assert list((root / name).iterdir()) == []
    assert (root / "user" / "default" / "workflows").is_dir()
    assert list((root / "user" / "default" / "workflows").iterdir()) == []


def test_shipped_placeholders_are_left_alone(tmp_path):
    """Deleting these would break ComfyUI's own examples without erasing anything."""
    root = _comfyui(tmp_path / "ComfyUI")
    (root / "input" / "example.png").write_bytes(b"shipped")
    (root / "input" / "3d").mkdir()
    (root / "output" / "_output_images_will_be_put_here").write_bytes(b"")
    (root / "input" / "mine.png").write_bytes(b"a" * 10)

    targets = collect_comfyui(root)
    collected = {path.name for target in targets for path in target.paths}
    assert collected == {"mine.png"}
    assert collected.isdisjoint(COMFYUI_KEEP)


def test_a_directory_that_is_not_comfyui_is_refused(tmp_path):
    """A mistyped variable must not be able to erase an arbitrary folder."""
    ordinary = tmp_path / "documents"
    ordinary.mkdir()
    (ordinary / "input").mkdir()
    assert not is_comfyui_install(ordinary)
    with pytest.raises(CleanupRefused, match="does not look like a ComfyUI installation"):
        collect_comfyui(ordinary)


def test_a_symlinked_entry_loses_only_the_link(tmp_path):
    """Unlinking a symlink never follows it, so what it points at is untouched."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "precious.png").write_bytes(b"keep me")
    root = _comfyui(tmp_path / "ComfyUI")
    (root / "temp" / "escape").symlink_to(outside)
    (root / "temp" / "real.png").write_bytes(b"z" * 10)

    targets = collect_comfyui(root)
    remove_paths([path for target in targets for path in target.paths])

    assert list((root / "temp").iterdir()) == []
    assert (outside / "precious.png").is_file()
    assert outside.is_dir()


def test_a_wipe_directory_that_is_itself_a_symlink_is_never_followed(tmp_path):
    """Resolving it would empty whatever it points at, which may be irreplaceable."""
    important = tmp_path / "important"
    important.mkdir()
    (important / "family_photo.png").write_bytes(b"irreplaceable")
    root = _comfyui(tmp_path / "ComfyUI")
    (root / "temp").rmdir()
    (root / "temp").symlink_to(important)

    targets = collect_comfyui(root)
    assert all("temp" not in target.label for target in targets)
    remove_paths([path for target in targets for path in target.paths])
    assert (important / "family_photo.png").read_bytes() == b"irreplaceable"

    assert skipped_symlinks(root) == [root / "temp"]


def test_the_weights_volume_is_never_a_target(tmp_path, monkeypatch):
    """Weights are models, not anything a user's picture went into."""
    work_root = tmp_path / "work"
    work_root.mkdir()
    _job_workspace(work_root, "aaaaaaaa-0000")
    monkeypatch.setenv("UPSCALER_WORK_ROOT", str(work_root))
    monkeypatch.delenv("UPSCALER_COMFYUI_INPUT_DIR", raising=False)
    monkeypatch.delenv("UPSCALER_COMFYUI_ROOT", raising=False)

    targets = collect(include_docker=False)
    assert str(work_root) in {target.detail for target in targets}
    assert all(target.volume != "upscaler_weights" for target in targets)
    # The only volume the collector will ever name is the job one.
    assert maintenance.DOCKER_JOB_VOLUME == "upscaler_work"


def test_the_running_app_is_detected_on_its_configured_port(monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        monkeypatch.setenv("UPSCALER_HOST", "127.0.0.1")
        monkeypatch.setenv("UPSCALER_PORT", str(port))
        assert maintenance.app_is_running()
    monkeypatch.setenv("UPSCALER_PORT", str(port))
    assert not maintenance.app_is_running()


def test_history_is_cleared_over_the_api_and_never_by_deleting_files(monkeypatch):
    """It lives only in ComfyUI's memory, so the API is the only way in."""
    called: list[str] = []
    monkeypatch.setattr(
        maintenance,
        "clear_comfyui_state",
        lambda base: called.append(base) or True,
    )
    problems = remove([Target(label="state", detail="u", endpoint="http://127.0.0.1:8188")])
    assert called == ["http://127.0.0.1:8188"]
    assert problems == []


def test_an_unreachable_comfyui_is_reported_rather_than_failing(monkeypatch):
    monkeypatch.setattr(maintenance, "clear_comfyui_state", lambda _base: False)
    problems = remove([Target(label="state", detail="u", endpoint="http://127.0.0.1:8188")])
    assert len(problems) == 1
    assert "already gone if it is not running" in problems[0]


@pytest.fixture(autouse=True)
def _no_recorded_install(tmp_path_factory, monkeypatch):
    """Keep these tests off whatever `make setup-comfyui` recorded on this machine.

    comfyui_root() falls back to that record, so without this a developer who has
    run the setup gets different results from one who has not.
    """
    monkeypatch.setattr(
        maintenance, "COMFYUI_RECORD", tmp_path_factory.mktemp("norecord") / "comfyui.conf"
    )


def test_the_recorded_installation_is_used_when_nothing_else_names_one(tmp_path, monkeypatch):
    """`make setup-comfyui` chose a path once; no later command should ask again."""
    monkeypatch.delenv("UPSCALER_COMFYUI_ROOT", raising=False)
    monkeypatch.delenv("UPSCALER_COMFYUI_INPUT_DIR", raising=False)
    root = _comfyui(tmp_path / "ComfyUI")
    record = tmp_path / "comfyui.conf"
    record.write_text(f"COMFYUI_ROOT={root}\nCOMFYUI_PORT=8188\n", encoding="utf-8")
    monkeypatch.setattr(maintenance, "COMFYUI_RECORD", record)

    assert comfyui_root() == root

    # An explicit path is for an installation this project never recorded.
    other = _comfyui(tmp_path / "Other")
    assert comfyui_root(str(other)) == other

    # A record naming a directory that has since gone is not a usable answer.
    record.write_text(f"COMFYUI_ROOT={tmp_path / 'gone'}\n", encoding="utf-8")
    assert comfyui_root() is None


def test_an_explicit_path_beats_an_environment_that_names_nothing(tmp_path, monkeypatch):
    """A shell running make does not carry the variables the app's launcher sets."""
    monkeypatch.delenv("UPSCALER_COMFYUI_ROOT", raising=False)
    monkeypatch.delenv("UPSCALER_COMFYUI_INPUT_DIR", raising=False)
    assert comfyui_root() is None

    root = _comfyui(tmp_path / "ComfyUI")
    assert comfyui_root(str(root)) == root


def test_the_input_directory_variable_still_locates_the_install(tmp_path, monkeypatch):
    root = _comfyui(tmp_path / "ComfyUI")
    monkeypatch.delenv("UPSCALER_COMFYUI_ROOT", raising=False)
    monkeypatch.setenv("UPSCALER_COMFYUI_INPUT_DIR", str(root / "input"))
    assert comfyui_root() == root


def test_without_a_comfyui_root_nothing_of_its_is_collected(tmp_path, monkeypatch):
    """The silent version of this let the command report success having skipped it."""
    work_root = tmp_path / "work"
    work_root.mkdir()
    _job_workspace(work_root, "aaaaaaaa-0000")
    monkeypatch.setenv("UPSCALER_WORK_ROOT", str(work_root))

    targets = collect(include_docker=False, root=None)
    assert [target.label for target in targets] == ["Job workspaces"]
    assert "NOT cleaned" in maintenance.NO_COMFYUI_NOTICE
