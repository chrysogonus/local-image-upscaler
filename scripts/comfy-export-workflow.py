#!/usr/bin/env python3
"""Convert a ComfyUI litegraph workflow into the API format ``POST /prompt`` needs.

ComfyUI saves workflows in its editor's own format: a ``nodes``/``links`` graph
where every node carries a positional ``widgets_values`` list. The API wants
``{node_id: {"class_type": ..., "inputs": {...}}}`` with inputs named. Only the
editor knows how to map one to the other, because the mapping lives in each
node's declared input order, which this script reads back from ``/object_info``.

Doing the conversion here rather than asking for a manual "Export (API)" click
keeps the checked-in templates reproducible: when a graph changes in ComfyUI,
re-run the same command and diff the result.

    scripts/comfy-export-workflow.py --input graph.json --output template.json

``--drop-node``, ``--rewire`` and ``--replace-class`` let one saved graph produce
a template that deliberately differs from it; every such difference has to be
spelled out on the command line, and the command lands in the template's header.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Types that appear as an editable widget rather than an input socket. Anything
# else is a connection, and never consumes a slot in ``widgets_values``.
WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})

# A combo whose selected option unfolds further widgets inline. The nested
# values are sent flattened as "<parent>.<child>".
DYNAMIC_COMBO_TYPE = "COMFY_DYNAMICCOMBO_V3"

# Options that make the editor add a second, unnamed widget straight after the
# input's own: the seed's "randomize" control and LoadImage's upload button.
# They hold editor state, carry no value the API accepts, and have to be skipped
# or every later widget on that node is read into the wrong input.
EXTRA_WIDGET_OPTIONS = ("control_after_generate", "image_upload")

# Editor furniture, not part of the graph.
NON_EXECUTING_TYPES = frozenset({"Note", "MarkdownNote", "Reroute", "PrimitiveNode"})

# litegraph node modes: 2 is muted, 4 is bypassed. Neither runs.
SKIPPED_MODES = frozenset({2, 4})


class ConversionError(RuntimeError):
    pass


def fetch_object_info(url: str) -> dict[str, Any]:
    endpoint = f"{url.rstrip('/')}/object_info"
    try:
        with urllib.request.urlopen(endpoint, timeout=30) as response:  # noqa: S310 - fixed scheme
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ConversionError(
            f"Could not read node definitions from {endpoint}: {exc}. "
            "ComfyUI has to be running for the conversion, since only it knows "
            "each node's input order."
        ) from exc


def _is_widget(spec: Any, options: dict[str, Any]) -> bool:
    if options.get("forceInput"):
        # Declared a widget type but drawn as a socket, so it has no widget value.
        return False
    if spec == DYNAMIC_COMBO_TYPE:
        return True
    if isinstance(spec, list):
        # A combo box: its options are listed inline.
        return True
    return spec in WIDGET_TYPES


def widget_entries(inputs_dict: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """Every widget-backed input as ``(name, spec, options)``, in editor order.

    Required inputs come before optional ones, which is the order the editor lays
    the widgets out in, and therefore the order ``widgets_values`` follows.
    """
    entries: list[tuple[str, Any, dict[str, Any]]] = []
    for section in ("required", "optional"):
        for name, entry in (inputs_dict.get(section) or {}).items():
            spec = entry[0] if isinstance(entry, list) and entry else entry
            options = entry[1] if isinstance(entry, list) and len(entry) > 1 else {}
            if not isinstance(options, dict):
                options = {}
            if _is_widget(spec, options):
                entries.append((name, spec, options))
    return entries


def assign_widgets(
    inputs_dict: dict[str, Any],
    values: list[Any],
    cursor: int = 0,
    prefix: str = "",
) -> tuple[dict[str, Any], int]:
    """Read the positional ``widgets_values`` list into named inputs.

    Returns the names it filled and how far it consumed, so a dynamic combo can
    recurse into whichever option the user picked and hand the cursor back.
    """
    assigned: dict[str, Any] = {}
    for name, spec, options in widget_entries(inputs_dict):
        if cursor >= len(values):
            # The editor omits trailing widgets it never drew; the node's own
            # default applies, so leaving the input out is correct.
            break
        full = f"{prefix}{name}"
        value = values[cursor]
        cursor += 1
        assigned[full] = value
        if spec == DYNAMIC_COMBO_TYPE:
            # Picking an option unfolds extra widgets inline. ComfyUI wants them
            # flattened next to the parent as "<parent>.<child>", which is what
            # its own frontend sends.
            selected = next(
                (o for o in options.get("options") or [] if o.get("key") == value), None
            )
            if selected is not None:
                nested, cursor = assign_widgets(
                    selected.get("inputs") or {}, values, cursor, f"{full}."
                )
                assigned.update(nested)
        else:
            cursor += sum(1 for flag in EXTRA_WIDGET_OPTIONS if options.get(flag))
    return assigned, cursor


def required_names(definition: dict[str, Any]) -> list[str]:
    return list((definition.get("input", {}) or {}).get("required", {}) or {})


def build_graph(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    dropped: set[int],
) -> dict[str, dict[str, Any]]:
    links: dict[int, tuple[int, int]] = {}
    for link in workflow.get("links") or []:
        # [link_id, origin_node, origin_slot, target_node, target_slot, type]
        links[link[0]] = (link[1], link[2])

    graph: dict[str, dict[str, Any]] = {}
    for node in workflow.get("nodes") or []:
        node_id = node["id"]
        class_type = node.get("type")
        if node_id in dropped or class_type in NON_EXECUTING_TYPES:
            continue
        if node.get("mode") in SKIPPED_MODES:
            continue
        definition = object_info.get(class_type)
        if definition is None:
            raise ConversionError(
                f"Node {node_id} is a {class_type!r}, which this ComfyUI does not have "
                "installed. Install the node pack it comes from, or drop the node."
            )

        values = list(node.get("widgets_values") or [])
        if isinstance(node.get("widgets_values"), dict):
            raise ConversionError(
                f"Node {node_id} ({class_type}) stores its widgets as a mapping, which this "
                "converter does not handle. Re-save the workflow from a current ComfyUI."
            )

        inputs, _ = assign_widgets(definition.get("input", {}) or {}, values)

        # Links win over widget values: an input converted to a socket keeps its
        # stale widget value in the saved graph, and sending that instead of the
        # connection would silently pin a size or seed the user thinks is wired.
        for slot in node.get("inputs") or []:
            link_id = slot.get("link")
            if link_id is None:
                continue
            origin = links.get(link_id)
            if origin is None:
                raise ConversionError(
                    f"Node {node_id} ({class_type}) input {slot.get('name')!r} references "
                    f"link {link_id}, which the workflow does not define."
                )
            inputs[slot["name"]] = [str(origin[0]), origin[1]]

        entry: dict[str, Any] = {"class_type": class_type, "inputs": inputs}
        title = node.get("title")
        if title:
            entry["_meta"] = {"title": title}
        graph[str(node_id)] = entry
    return graph


def apply_rewires(graph: dict[str, dict[str, Any]], rewires: list[str]) -> None:
    for rewire in rewires:
        try:
            target, source = rewire.split("=", 1)
            node_id, input_name = target.split(".", 1)
            source_node, source_slot = source.split(":", 1)
            slot_index = int(source_slot)
        except ValueError as exc:
            raise ConversionError(
                f"--rewire {rewire!r} is not in NODE.input=SOURCE:SLOT form."
            ) from exc
        if node_id not in graph:
            raise ConversionError(f"--rewire {rewire!r} targets node {node_id}, which is not kept.")
        if source_node not in graph:
            raise ConversionError(
                f"--rewire {rewire!r} sources node {source_node}, which is not kept."
            )
        graph[node_id]["inputs"][input_name] = [source_node, slot_index]


def apply_class_replacements(
    graph: dict[str, dict[str, Any]],
    object_info: dict[str, Any],
    replacements: list[str],
) -> None:
    for replacement in replacements:
        try:
            node_id, new_class = replacement.split("=", 1)
        except ValueError as exc:
            raise ConversionError(
                f"--replace-class {replacement!r} is not in NODE=ClassType form."
            ) from exc
        if node_id not in graph:
            raise ConversionError(
                f"--replace-class {replacement!r} targets node {node_id}, which is not kept."
            )
        definition = object_info.get(new_class)
        if definition is None:
            raise ConversionError(
                f"--replace-class {replacement!r} names {new_class!r}, which this ComfyUI "
                "does not have installed."
            )
        keep = set(required_names(definition)) | {
            name for name, _, _ in widget_entries(definition.get("input", {}) or {})
        }
        node = graph[node_id]
        node["class_type"] = new_class
        node["inputs"] = {k: v for k, v in node["inputs"].items() if k in keep}


def validate(graph: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> None:
    problems: list[str] = []
    for node_id, node in graph.items():
        definition = object_info[node["class_type"]]
        for name in required_names(definition):
            if name not in node["inputs"]:
                problems.append(
                    f"node {node_id} ({node['class_type']}) is missing required input {name!r}"
                )
        for name, value in node["inputs"].items():
            wired = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            if wired and value[0] not in graph:
                problems.append(
                    f"node {node_id} ({node['class_type']}) input {name!r} is wired to "
                    f"node {value[0]}, which is not in the template"
                )
    if problems:
        raise ConversionError(
            "The converted graph would be rejected by ComfyUI:\n  - " + "\n  - ".join(problems)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path, help="saved litegraph workflow JSON")
    parser.add_argument("--output", required=True, type=Path, help="API-format template to write")
    parser.add_argument("--url", default="http://127.0.0.1:8188", help="running ComfyUI base URL")
    parser.add_argument(
        "--drop-node",
        action="append",
        default=[],
        help="node id to leave out; repeatable, or comma-separated",
    )
    parser.add_argument(
        "--rewire",
        action="append",
        default=[],
        help="NODE.input=SOURCE:SLOT, applied after dropping; repeatable",
    )
    parser.add_argument(
        "--replace-class",
        action="append",
        default=[],
        help="NODE=ClassType, keeping only inputs the new class declares; repeatable",
    )
    args = parser.parse_args(argv)

    dropped: set[int] = set()
    for item in args.drop_node:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                dropped.add(int(part))

    try:
        workflow = json.loads(args.input.read_text())
        object_info = fetch_object_info(args.url)
        graph = build_graph(workflow, object_info, dropped)
        apply_rewires(graph, args.rewire)
        apply_class_replacements(graph, object_info, args.replace_class)
        validate(graph, object_info)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(graph)} nodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
