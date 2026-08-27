"""Small declarative builder for openable ComfyUI workflow files.

The API graph ComfyUI executes is easy to read, but its editor format also
needs widget arrays, socket indices, and globally numbered links. Keeping that
bookkeeping here lets authored workflows describe only nodes and connections.
"""

from __future__ import annotations

from typing import Any

NodeSpec = tuple[int, str, str, dict[str, tuple[int, int]], dict[str, Any]]

WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
EXTRA_WIDGETS = ("control_after_generate", "image_upload")


def _entries(definition: dict[str, Any], section: str) -> list[tuple[str, Any, dict[str, Any]]]:
    entries = []
    for name, entry in ((definition.get("input") or {}).get(section) or {}).items():
        spec = entry[0] if isinstance(entry, list) and entry else entry
        options = entry[1] if isinstance(entry, list) and len(entry) > 1 else {}
        entries.append((name, spec, options if isinstance(options, dict) else {}))
    return entries


def _is_widget(spec: Any, options: dict[str, Any]) -> bool:
    if options.get("forceInput"):
        return False
    return spec == DYNAMIC_COMBO or isinstance(spec, list) or spec in WIDGET_TYPES


def _default_for(spec: Any, options: dict[str, Any]) -> Any:
    if "default" in options:
        return options["default"]
    if isinstance(spec, list) and spec:
        return spec[0]
    return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(spec)


def build_workflow(
    graph: list[NodeSpec], object_info: dict[str, Any], workflow_id: str
) -> dict[str, Any]:
    """Build ComfyUI's editor format from a topologically ordered node list."""
    depth: dict[int, int] = {}
    for node_id, _, _, links, _ in graph:
        depth[node_id] = 0 if not links else 1 + max(depth[source] for source, _ in links.values())

    column_counts: dict[int, int] = {}
    outgoing: dict[tuple[int, int], list[int]] = {}
    nodes: list[dict[str, Any]] = []
    workflow_links: list[list[Any]] = []
    link_id = 0

    for node_id, class_type, title, links, widgets in graph:
        definition = object_info.get(class_type)
        if definition is None:
            raise SystemExit(f"error: this ComfyUI has no {class_type!r} node")
        all_entries = _entries(definition, "required") + _entries(definition, "optional")

        values: list[Any] = []
        input_slots: list[dict[str, Any]] = []
        for name, spec, options in all_entries:
            if _is_widget(spec, options):
                values.append(widgets.get(name, _default_for(spec, options)))
                for flag in EXTRA_WIDGETS:
                    if options.get(flag):
                        values.append("randomize" if flag == "control_after_generate" else "image")
                if name not in links:
                    continue
            input_slots.append({"name": name, "type": spec if isinstance(spec, str) else "COMBO"})

        for index, slot in enumerate(input_slots):
            source = links.get(slot["name"])
            if source is None:
                slot["link"] = None
                continue
            link_id += 1
            slot["link"] = link_id
            workflow_links.append([link_id, source[0], source[1], node_id, index, slot["type"]])
            outgoing.setdefault(source, []).append(link_id)

        column = depth[node_id]
        row = column_counts.get(column, 0)
        column_counts[column] = row + 1
        node: dict[str, Any] = {
            "id": node_id,
            "type": class_type,
            "pos": [column * 340, row * 190],
            "size": [300, 100],
            "flags": {},
            "order": len(nodes),
            "mode": 0,
            "inputs": input_slots,
            "outputs": [
                {"name": name, "type": kind, "links": None, "slot_index": index}
                for index, (name, kind) in enumerate(
                    zip(
                        definition.get("output_name") or definition.get("output") or [],
                        definition.get("output") or [],
                        strict=False,
                    )
                )
            ],
            "properties": {"Node name for S&R": class_type},
            "widgets_values": values,
        }
        if title:
            node["title"] = title
        nodes.append(node)

    for node in nodes:
        for slot in node["outputs"]:
            slot["links"] = outgoing.get((node["id"], slot["slot_index"]), [])

    return {
        "id": workflow_id,
        "revision": 0,
        "last_node_id": max(node_id for node_id, *_ in graph),
        "last_link_id": link_id,
        "nodes": nodes,
        "links": workflow_links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }
