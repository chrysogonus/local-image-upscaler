from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY_ROOT / "models" / "manifest.json"
WORKFLOWS = REPOSITORY_ROOT / "backend" / "upscaler" / "workflows"
SCRIPTS = REPOSITORY_ROOT / "scripts"
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_script(filename: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            filename.removesuffix(".py"), SCRIPTS / filename
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_model_manifest_has_complete_immutable_download_metadata() -> None:
    manifest = load_json(MANIFEST)
    assert manifest["schema_version"] == 3
    entries = manifest["runtimes"] + manifest["weights"] + manifest["repositories"]
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)), "manifest ids must be unique"

    for entry in entries:
        for field in ("id", "name", "license", "homepage", "notes"):
            assert isinstance(entry.get(field), str) and entry[field].strip(), (
                f"{entry.get('id', '<unknown>')} has no {field}"
            )
        assert entry["homepage"].startswith("https://")

    for entry in manifest["runtimes"]:
        assert entry["archive_url"].startswith("https://")
        assert SHA256.fullmatch(entry["sha256"])
        assert entry["platform"] != "any"

    for entry in manifest["weights"]:
        assert entry["url"].startswith("https://")
        assert SHA256.fullmatch(entry["sha256"])
        assert Path(entry["filename"]).name == entry["filename"]
        assert entry["group"]
        if entry.get("noncommercial"):
            assert "commercial" in entry["license"].lower() or "s-lab" in entry["license"].lower()

    for entry in manifest["repositories"]:
        assert entry["group"]
        assert "/" in entry["repo_id"]
        assert Path(entry["directory"]).name == entry["directory"]
        assert entry["allow_patterns"]
        revision = entry.get("revision")
        assert (isinstance(revision, str) and COMMIT.fullmatch(revision)) or (
            revision is None and entry.get("gated") is True
        ), f"{entry['id']} must be commit-pinned unless it is gated"


# The sentence NOTICE carries while nothing installed is restricted. NOTICE is
# the legal artifact - it ships in the wheel and at /usr/share/licenses in the
# container - so it is the one file that must not describe a regime the
# manifest does not have.
NO_RESTRICTED_WEIGHTS_CLAIM = "None carries a noncommercial restriction."


def test_notice_describes_the_licences_the_manifest_actually_installs() -> None:
    """Keep the legal notice and the manifest from telling different stories.

    An earlier NOTICE claimed some installed weights forbade commercial use and
    that the application labelled their results. Neither was true: every entry
    is permissive, and no product code reads ``noncommercial``. Nothing else in
    the suite reads NOTICE, so a contradiction there is invisible until someone
    quotes it back.
    """
    manifest = load_json(MANIFEST)
    weights = manifest["runtimes"] + manifest["weights"] + manifest["repositories"]
    restricted = sorted(entry["id"] for entry in weights if entry.get("noncommercial"))
    notice = (REPOSITORY_ROOT / "NOTICE").read_text(encoding="utf-8")

    if restricted:
        assert NO_RESTRICTED_WEIGHTS_CLAIM not in notice, (
            f"NOTICE says nothing is restricted, but {', '.join(restricted)} is."
        )
        for entry_id in restricted:
            assert entry_id in notice, f"NOTICE does not name the restricted weight {entry_id}."
    else:
        assert NO_RESTRICTED_WEIGHTS_CLAIM in notice, (
            "No manifest entry is noncommercial, so NOTICE has to say so plainly."
        )
        # The claim that used to be wrong, in the present tense that made it a
        # statement about this tree rather than about a possible future one.
        assert "Some of those licences forbid commercial use." not in notice


def test_base_image_digests_are_pinned_only_in_the_dockerfile() -> None:
    """One place per pin, so Dependabot's Docker updater reaches all of them.

    The CUDA digest was once restated in the Compose overlay, in CI, and in the
    deployment guide. Dependabot only maintains digests on a Dockerfile FROM
    line, so those three drifted by design rather than by accident, and a manual
    bump that missed one left CI validating an image nobody ran.
    """
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    pinned = set(re.findall(r"@sha256:([0-9a-f]{64})", dockerfile))
    assert pinned, "the Dockerfile pins no image digests at all"

    others = [
        path
        for path in (
            REPOSITORY_ROOT / "docker-compose.yml",
            REPOSITORY_ROOT / "docker-compose.cuda.yml",
            REPOSITORY_ROOT / "docker-compose.comfyui.yml",
            REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml",
            *sorted((REPOSITORY_ROOT / "docs").glob("*.md")),
        )
        if path.is_file()
    ]
    restated = [
        f"{path.relative_to(REPOSITORY_ROOT)} restates {digest[:12]}"
        for path in others
        for digest in re.findall(r"@sha256:([0-9a-f]{64})", path.read_text(encoding="utf-8"))
        if digest in pinned
    ]
    assert not restated, "\n".join(restated)


@pytest.mark.parametrize(
    ("workflow_id", "builder_name"),
    [
        ("illustration-upscale", "build-illustration-workflow.py"),
    ],
)
def test_authored_workflows_match_their_builders(workflow_id: str, builder_name: str) -> None:
    """Catch hand edits to either generated representation of locally authored graphs."""
    graph = load_script(builder_name).GRAPH
    source = load_json(WORKFLOWS / "source" / f"{workflow_id}.workflow.json")
    api = load_json(WORKFLOWS / f"{workflow_id}.api.json")
    source_nodes = {node["id"]: node for node in source["nodes"]}
    expected_ids = {node_id for node_id, *_ in graph}

    assert set(source_nodes) == expected_ids
    assert set(api) == {str(node_id) for node_id in expected_ids}

    source_links = {link[0]: link for link in source["links"]}
    for node_id, class_type, title, links, widgets in graph:
        saved = source_nodes[node_id]
        exported = api[str(node_id)]
        assert saved["type"] == exported["class_type"] == class_type
        assert saved.get("title", "") == title
        assert exported.get("_meta", {}).get("title", "") == title
        for name, value in widgets.items():
            assert exported["inputs"][name] == value
        for name, (origin, slot) in links.items():
            assert exported["inputs"][name] == [str(origin), slot]
            source_input = next(item for item in saved["inputs"] if item["name"] == name)
            link = source_links[source_input["link"]]
            assert (link[1], link[2], link[3]) == (origin, slot, node_id)


# The engines whose inference actually averages the eight orientations. The
# resampler has no inference to average, and ComfyUI runs a graph that controls
# its own, so both discard the request.
AUGMENTING_ENGINES = {"realesrgan_cuda.py", "realesrgan_ncnn.py", "spandrel_sr.py"}


def test_only_engines_that_run_augmentation_claim_to_support_it() -> None:
    """``supports_tta`` is what the interface offers, so it has to be the truth.

    The control costs eight inferences per pass. An engine that never reads
    ``request.tta`` would charge that for an identical result, which is why the
    claim is checked against the adapters rather than trusted.
    """
    adapters = REPOSITORY_ROOT / "backend" / "upscaler" / "models"
    declared = 0
    for module in sorted(adapters.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        if "native_scales = " not in source:
            continue
        declared += 1
        assert "supports_tta = " in source, f"{module.name} declares no augmentation support."
        claims = "supports_tta = True" in source
        assert claims == (module.name in AUGMENTING_ENGINES), (
            f"{module.name} claims supports_tta={claims}, which contradicts the reviewed list."
        )
        if claims:
            assert "request.tta" in source, f"{module.name} claims support it never reads."
    assert declared == 5


def markdown_files() -> list[Path]:
    """Every hand-written Markdown file whose links are worth enforcing.

    `.github/` is included because the issue and pull-request templates point
    readers at the same manual the rest of the prose does, and a template is the
    one file nobody re-reads until it is already in front of a contributor.
    """
    return (
        sorted(REPOSITORY_ROOT.glob("*.md"))
        + sorted((REPOSITORY_ROOT / "docs").glob("*.md"))
        + sorted((REPOSITORY_ROOT / ".github").rglob("*.md"))
    )


def heading_slug(heading: str) -> str:
    """Reproduce GitHub's anchor for a heading.

    github-slugger lowercases, drops everything that is not a word character,
    space, or hyphen, then replaces spaces one at a time. The last part matters:
    an em dash between two words leaves two spaces behind and therefore a double
    hyphen, which is why `#comfyui--run-your-own-workflows` is spelled that way.
    """
    cleaned = re.sub(r"[^\w\s-]", "", heading.strip().lower(), flags=re.UNICODE)
    return cleaned.replace(" ", "-")


def test_documentation_links_and_anchors_resolve() -> None:
    """The manual is split across `docs/`, so its cross-links are load-bearing.

    Moving a section is a rename of every anchor pointing at it, and a broken
    one is invisible until a reader follows it. Nothing else in the suite reads
    the prose, so this is what keeps the split honest.
    """
    anchors = {
        path: {
            heading_slug(match.group(2))
            for line in path.read_text(encoding="utf-8").split("\n")
            if (match := re.match(r"^(#{1,6})\s+(.*)$", line))
        }
        for path in markdown_files()
    }

    broken: list[str] = []
    checked = 0
    for path in markdown_files():
        body = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", body):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            relative, _, anchor = target.partition("#")
            destination = (path.parent / relative) if relative else path
            if relative and not destination.exists():
                broken.append(f"{path.name}: no such file -> {target}")
                continue
            if anchor and anchor not in anchors.get(destination.resolve(), set()):
                broken.append(f"{path.name}: no such anchor -> {target}")

    assert not broken, "\n".join(broken)
    assert checked >= 40, f"only {checked} links checked; the documentation lost its wiring."
