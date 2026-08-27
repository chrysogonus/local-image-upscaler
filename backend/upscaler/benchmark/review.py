# ruff: noqa: E501
from __future__ import annotations

import hashlib
import html
import itertools
import json
import os
import random
import shutil
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import load_prepared
from .runner import CANDIDATE_NAMES, load_run

REVIEW_SCHEMA_VERSION = 1
ARTIFACT_TAGS = ("halos", "oversmoothing", "invented-texture", "color-shift", "seams")
CHOICES = {"a", "b", "tie", "cannot"}


class BenchmarkReviewError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_token(run_id: str, output: dict[str, Any]) -> str:
    value = (f"{run_id}:{output['case_id']}:{output['candidate_id']}:{output['sha256']}").encode()
    return hashlib.sha256(value).hexdigest()


def _relative_uri(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _prepared_for_run(run_dir: Path, run: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    relative = run.get("dataset", {}).get("prepared_manifest")
    if not isinstance(relative, str):
        raise BenchmarkReviewError("run does not name its prepared dataset")
    path = (run_dir / relative).resolve()
    prepared = load_prepared(path)
    if prepared.get("dataset_digest") != run.get("dataset", {}).get("digest"):
        raise BenchmarkReviewError("prepared dataset does not match this run")
    if hashlib.sha256(path.read_bytes()).hexdigest() != run.get("dataset", {}).get(
        "prepared_digest"
    ):
        raise BenchmarkReviewError("materialized benchmark inputs changed after the run")
    return path, prepared


REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind upscaler review</title>
  <style>
    :root { color-scheme: dark; font: 16px/1.45 system-ui, sans-serif; background:#111; color:#eee; }
    * { box-sizing: border-box; }
    body { margin:0; }
    button, input { font:inherit; }
    button { color:inherit; background:#292929; border:1px solid #666; border-radius:.4rem; padding:.55rem .8rem; }
    button:hover, button:focus-visible { border-color:#fff; outline:2px solid transparent; }
    button.primary { background:#315f50; }
    header { position:sticky; top:0; z-index:5; display:flex; align-items:center; gap:1rem; padding:.65rem 1rem; background:#171717ee; border-bottom:1px solid #444; }
    header h1 { font-size:1.05rem; margin:0 auto 0 0; }
    main { max-width:1500px; margin:auto; padding:1rem; }
    .context { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin-bottom:1rem; }
    .context figure { margin:0; min-width:0; }
    .context img { width:100%; height:180px; object-fit:contain; background:#080808; border:1px solid #444; }
    figcaption { color:#bbb; margin-top:.25rem; }
    .compare { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; }
    .candidate h2 { margin:.25rem 0; font-size:1rem; text-align:center; }
    .viewport { position:relative; overflow:hidden; height:min(58vh,620px); background:#080808; border:1px solid #555; touch-action:none; cursor:grab; }
    .viewport:active { cursor:grabbing; }
    .viewport img { position:absolute; left:0; top:0; max-width:none; user-select:none; pointer-events:none; transform-origin:0 0; }
    .tools, .judgments, .navigation { display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:.5rem; margin-top:.8rem; }
    .tools output { min-width:4rem; }
    .artifacts { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1rem; margin-top:1rem; }
    fieldset { border:1px solid #444; border-radius:.4rem; }
    fieldset label { display:inline-flex; align-items:center; gap:.3rem; margin:.25rem .6rem .25rem 0; }
    .status { color:#bbb; }
    .choice.selected { background:#315f50; border-color:#9ed3c0; }
    .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
    @media (max-width:800px) { .compare, .context, .artifacts { grid-template-columns:1fr; } .viewport { height:42vh; } }
    @media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; transition:none !important; } }
  </style>
</head>
<body>
  <header>
    <h1>Blind perceptual review</h1>
    <span id="progress" class="status"></span>
    <button id="export" type="button">Export completed session</button>
  </header>
  <main>
    <p id="case-meta" class="status"></p>
    <section class="context" aria-label="Source context">
      <figure><img id="source" alt="Low-resolution source"><figcaption>Source</figcaption></figure>
      <figure id="reference-figure"><img id="reference" alt="High-resolution reference"><figcaption>Reference</figcaption></figure>
    </section>
    <section class="compare" aria-label="Anonymous output comparison">
      <div class="candidate"><h2>Output A</h2><div class="viewport"><img id="image-a" alt="Anonymous candidate output A"></div></div>
      <div class="candidate"><h2>Output B</h2><div class="viewport"><img id="image-b" alt="Anonymous candidate output B"></div></div>
    </section>
    <div class="tools" aria-label="Image navigation">
      <button id="fit" type="button">Fit</button><button id="actual" type="button">1:1</button>
      <label>Zoom <input id="zoom" type="range" min="10" max="300" value="100"><output id="zoom-value">100%</output></label>
    </div>
    <div class="judgments" role="group" aria-label="Which output is better overall?">
      <button class="choice" data-choice="a" type="button">A is better <kbd>A</kbd></button>
      <button class="choice" data-choice="tie" type="button">Tie <kbd>T</kbd></button>
      <button class="choice" data-choice="b" type="button">B is better <kbd>B</kbd></button>
      <button class="choice" data-choice="cannot" type="button">Cannot judge <kbd>X</kbd></button>
    </div>
    <section class="artifacts" aria-label="Optional visible artifacts">
      <fieldset id="artifacts-a"><legend>Artifacts in A (optional)</legend></fieldset>
      <fieldset id="artifacts-b"><legend>Artifacts in B (optional)</legend></fieldset>
    </section>
    <div class="navigation"><button id="previous" type="button">Previous</button><button id="next" class="primary" type="button">Next</button></div>
    <p class="status">Keys: A/B/T/X records a judgment, arrow keys move between pairs. Drag either output to pan both; use the wheel or slider to zoom at matched coordinates.</p>
    <p id="announcement" class="sr-only" aria-live="polite"></p>
  </main>
  <script id="review-data" type="application/json">__PAYLOAD__</script>
  <script>
  (() => {
    const data = JSON.parse(document.getElementById('review-data').textContent);
    const storageKey = `upscaler-benchmark:${data.run_id}:${data.session_id}`;
    let saved;
    try { saved = JSON.parse(localStorage.getItem(storageKey) || '{}'); } catch { saved = {}; }
    const state = { index: Number.isInteger(saved.index) ? saved.index : 0, answers: saved.answers || {}, scale:1, x:0, y:0, fit:true };
    const byId = id => document.getElementById(id);
    const imageA=byId('image-a'), imageB=byId('image-b');
    const panes=[imageA.parentElement,imageB.parentElement];
    const artifactLabels = {halos:'Halos',oversmoothing:'Oversmoothing','invented-texture':'Invented texture','color-shift':'Color shift',seams:'Tile seams'};
    for (const side of ['a','b']) {
      const field=byId(`artifacts-${side}`);
      for (const tag of data.artifact_tags) {
        const label=document.createElement('label');
        label.innerHTML=`<input type="checkbox" value="${tag}"> ${artifactLabels[tag]}`;
        label.querySelector('input').addEventListener('change', saveArtifacts);
        field.append(label);
      }
    }
    function keyFor(item) { return `${item.case_id}:${item.pair_id}`; }
    function persist() { localStorage.setItem(storageKey, JSON.stringify({index:state.index,answers:state.answers})); }
    function current() { return data.comparisons[state.index]; }
    function applyTransform() {
      const transform=`translate(${state.x}px,${state.y}px) scale(${state.scale})`;
      imageA.style.transform=transform; imageB.style.transform=transform;
      byId('zoom').value=String(Math.round(state.scale*100)); byId('zoom-value').value=`${Math.round(state.scale*100)}%`;
    }
    function fit() {
      if (!imageA.naturalWidth) return;
      state.scale=Math.min(panes[0].clientWidth/imageA.naturalWidth, panes[0].clientHeight/imageA.naturalHeight);
      state.x=(panes[0].clientWidth-imageA.naturalWidth*state.scale)/2;
      state.y=(panes[0].clientHeight-imageA.naturalHeight*state.scale)/2; state.fit=true; applyTransform();
    }
    function actual() { state.scale=1; state.x=0; state.y=0; state.fit=false; applyTransform(); }
    function answer() { return state.answers[keyFor(current())] || {choice:null,tags_a:[],tags_b:[]}; }
    function saveArtifacts() {
      const value=answer();
      for (const side of ['a','b']) value[`tags_${side}`]=[...byId(`artifacts-${side}`).querySelectorAll('input:checked')].map(x=>x.value);
      state.answers[keyFor(current())]=value; persist();
    }
    function choose(choice) {
      const value=answer(); value.choice=choice; state.answers[keyFor(current())]=value; persist();
      byId('announcement').textContent=`Recorded ${choice}`; renderControls();
    }
    function renderControls() {
      const value=answer();
      document.querySelectorAll('.choice').forEach(button=>button.classList.toggle('selected',button.dataset.choice===value.choice));
      for (const side of ['a','b']) byId(`artifacts-${side}`).querySelectorAll('input').forEach(input=>input.checked=value[`tags_${side}`].includes(input.value));
      const count=Object.values(state.answers).filter(value=>value.choice).length;
      byId('progress').textContent=`${count} of ${data.comparisons.length} judged`;
      byId('previous').disabled=state.index===0; byId('next').disabled=state.index===data.comparisons.length-1;
    }
    function render() {
      const item=current(); state.fit=true;
      byId('case-meta').textContent=`${state.index+1}/${data.comparisons.length} · ${item.title} · ${item.track} · ${item.tags.join(', ')}`;
      byId('source').src=item.source; byId('reference-figure').hidden=!item.reference;
      if (item.reference) byId('reference').src=item.reference;
      imageA.src=item.a.src; imageB.src=item.b.src; renderControls(); persist();
    }
    function move(delta) { state.index=Math.max(0,Math.min(data.comparisons.length-1,state.index+delta)); render(); }
    document.querySelectorAll('.choice').forEach(button=>button.addEventListener('click',()=>choose(button.dataset.choice)));
    byId('previous').addEventListener('click',()=>move(-1)); byId('next').addEventListener('click',()=>move(1));
    byId('fit').addEventListener('click',fit); byId('actual').addEventListener('click',actual);
    byId('zoom').addEventListener('input',event=>{state.scale=Number(event.target.value)/100;state.fit=false;applyTransform();});
    let drag=null;
    for (const pane of panes) {
      pane.addEventListener('pointerdown',event=>{drag={clientX:event.clientX,clientY:event.clientY,x:state.x,y:state.y};pane.setPointerCapture(event.pointerId);});
      pane.addEventListener('pointermove',event=>{if(!drag)return;state.x=drag.x+event.clientX-drag.clientX;state.y=drag.y+event.clientY-drag.clientY;state.fit=false;applyTransform();});
      pane.addEventListener('pointerup',()=>{drag=null;});
      pane.addEventListener('wheel',event=>{event.preventDefault();state.scale=Math.max(.1,Math.min(3,state.scale*(event.deltaY<0?1.1:.9)));state.fit=false;applyTransform();},{passive:false});
    }
    imageA.addEventListener('load',()=>{if(state.fit)fit();}); window.addEventListener('resize',()=>{if(state.fit)fit();});
    document.addEventListener('keydown',event=>{
      if (event.target.matches('input')) return;
      const key=event.key.toLowerCase();
      if(key==='a'||key==='b'||key==='t'||key==='x') { choose({a:'a',b:'b',t:'tie',x:'cannot'}[key]); event.preventDefault(); }
      else if(event.key==='ArrowLeft') {move(-1);event.preventDefault();} else if(event.key==='ArrowRight'){move(1);event.preventDefault();}
    });
    byId('export').addEventListener('click',()=>{
      const judgments=[];
      for(const item of data.comparisons){const value=state.answers[keyFor(item)];if(!value||!value.choice)continue;judgments.push({case_id:item.case_id,left_output_token:item.a.token,right_output_token:item.b.token,left_output_hash:item.a.hash,right_output_hash:item.b.hash,choice:value.choice,artifacts:{left:value.tags_a,right:value.tags_b}});}
      if(judgments.length!==data.comparisons.length){alert('Judge every pair before exporting. Use Cannot judge when neither output can be assessed.');return;}
      const session={schema_version:data.schema_version,run_id:data.run_id,session_id:data.session_id,created_at:data.created_at,completed_at:new Date().toISOString(),judgments};
      const blob=new Blob([JSON.stringify(session,null,2)+'\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=`review-${data.run_id}-${data.session_id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
    });
    render();
  })();
  </script>
</body>
</html>
"""


def generate_review(run_path: Path, *, session_id: str | None = None) -> Path:
    run_dir = run_path if run_path.is_dir() else run_path.parent
    run = load_run(run_dir)
    prepared_path, prepared = _prepared_for_run(run_dir, run)
    prepared_root = prepared_path.parent
    session_id = session_id or str(uuid.uuid4())
    try:
        uuid.UUID(session_id)
    except ValueError as exc:
        raise BenchmarkReviewError("session id must be a UUID") from exc
    output_by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for output in run["outputs"]:
        output_by_case[output["case_id"]][output["candidate_id"]] = output
    prepared_by_case = {case["id"]: case for case in prepared["cases"]}
    asset_root = run_dir / "review-assets" / session_id
    comparisons = []
    generator = random.Random(session_id)
    pair_offsets: dict[tuple[str, str], int] = {}
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for case in run["cases"]:
        prepared_case = prepared_by_case[case["id"]]
        source = prepared_root / prepared_case["input"]
        source_asset = asset_root / f"source-{case['id']}{source.suffix.lower()}"
        _link_or_copy(source, source_asset)
        reference_asset = None
        if prepared_case["reference"]:
            reference = prepared_root / prepared_case["reference"]
            reference_asset = asset_root / f"reference-{case['id']}.png"
            _link_or_copy(reference, reference_asset)
        candidates = sorted(output_by_case[case["id"]])
        for first, second in itertools.combinations(candidates, 2):
            pair = (first, second)
            offset = pair_offsets.setdefault(pair, generator.randrange(2))
            swap = (pair_counts[pair] + offset) % 2
            pair_counts[pair] += 1
            left, right = (second, first) if swap else (first, second)
            sides = []
            for candidate in (left, right):
                output = output_by_case[case["id"]][candidate]
                source_output = run_dir / output["path"]
                alias = asset_root / f"{output['sha256'][:24]}.png"
                _link_or_copy(source_output, alias)
                sides.append(
                    {
                        "hash": output["sha256"],
                        "token": output_token(run["run_id"], output),
                        "src": _relative_uri(alias, run_dir / "reviews"),
                    }
                )
            pair_digest = hashlib.sha256(f"{case['id']}:{first}:{second}".encode()).hexdigest()[:16]
            comparisons.append(
                {
                    "case_id": case["id"],
                    "pair_id": pair_digest,
                    "title": case["title"],
                    "track": case["track"],
                    "tags": case["tags"],
                    "source": _relative_uri(source_asset, run_dir / "reviews"),
                    "reference": (
                        _relative_uri(reference_asset, run_dir / "reviews")
                        if reference_asset
                        else None
                    ),
                    "a": sides[0],
                    "b": sides[1],
                }
            )
    generator.shuffle(comparisons)
    payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "run_id": run["run_id"],
        "session_id": session_id,
        "created_at": _utc_now(),
        "artifact_tags": list(ARTIFACT_TAGS),
        "comparisons": comparisons,
    }
    encoded = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    review_path = run_dir / "reviews" / f"review-{session_id}.html"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(REVIEW_HTML.replace("__PAYLOAD__", encoded), encoding="utf-8")
    return review_path


def _load_session(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkReviewError(f"could not read review session {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise BenchmarkReviewError(f"{path} uses an unsupported review schema")
    return payload


def validate_session(run: dict[str, Any], session: dict[str, Any]) -> list[dict[str, Any]]:
    if session.get("run_id") != run.get("run_id"):
        raise BenchmarkReviewError("review session belongs to another benchmark run")
    try:
        uuid.UUID(str(session.get("session_id")))
    except ValueError as exc:
        raise BenchmarkReviewError("review session id must be a UUID") from exc
    output_by_token = {output_token(run["run_id"], output): output for output in run["outputs"]}
    expected_pairs = {
        (case["id"], frozenset(pair))
        for case in run["cases"]
        for pair in itertools.combinations(CANDIDATE_NAMES, 2)
    }
    judgments = session.get("judgments")
    if not isinstance(judgments, list):
        raise BenchmarkReviewError("review judgments must be a list")
    seen: set[tuple[str, frozenset[str]]] = set()
    validated = []
    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise BenchmarkReviewError("every judgment must be an object")
        case_id = judgment.get("case_id")
        left_token = judgment.get("left_output_token")
        right_token = judgment.get("right_output_token")
        if not isinstance(left_token, str) or not isinstance(right_token, str):
            raise BenchmarkReviewError("judgment contains an unknown output token")
        left = output_by_token.get(left_token)
        right = output_by_token.get(right_token)
        if left is None or right is None or left is right:
            raise BenchmarkReviewError("judgment contains an unknown or duplicate output token")
        if (
            judgment.get("left_output_hash") != left["sha256"]
            or judgment.get("right_output_hash") != right["sha256"]
        ):
            raise BenchmarkReviewError("judgment output hash was changed")
        if left["case_id"] != case_id or right["case_id"] != case_id:
            raise BenchmarkReviewError("judgment output does not belong to its case")
        pair_key = (case_id, frozenset((left["candidate_id"], right["candidate_id"])))
        if pair_key in seen:
            raise BenchmarkReviewError("review session contains a duplicate candidate pair")
        seen.add(pair_key)
        choice = judgment.get("choice")
        if choice not in CHOICES:
            raise BenchmarkReviewError(f"unknown review choice {choice!r}")
        artifacts = judgment.get("artifacts")
        if not isinstance(artifacts, dict):
            raise BenchmarkReviewError("judgment artifacts must be an object")
        for side in ("left", "right"):
            tags = artifacts.get(side)
            if not isinstance(tags, list) or any(tag not in ARTIFACT_TAGS for tag in tags):
                raise BenchmarkReviewError(f"invalid {side} artifact tags")
            if len(tags) != len(set(tags)):
                raise BenchmarkReviewError(f"duplicate {side} artifact tag")
        validated.append(
            {
                "case_id": case_id,
                "left": left,
                "right": right,
                "choice": choice,
                "artifacts": artifacts,
            }
        )
    if seen != expected_pairs:
        raise BenchmarkReviewError(
            f"review session is incomplete: found {len(seen)} of {len(expected_pairs)} pairs"
        )
    return validated


def _score_rows(
    candidates: list[str],
    judgments: list[dict[str, Any]],
    include: Any,
) -> list[dict[str, Any]]:
    totals = {candidate: {"points": 0.0, "judgments": 0} for candidate in candidates}
    for judgment in judgments:
        if not include(judgment) or judgment["choice"] == "cannot":
            continue
        left = judgment["left"]["candidate_id"]
        right = judgment["right"]["candidate_id"]
        totals[left]["judgments"] += 1
        totals[right]["judgments"] += 1
        if judgment["choice"] == "tie":
            totals[left]["points"] += 0.5
            totals[right]["points"] += 0.5
        elif judgment["choice"] == "a":
            totals[left]["points"] += 1.0
        else:
            totals[right]["points"] += 1.0
    rows = []
    for candidate in candidates:
        total = totals[candidate]
        judgments_count = total["judgments"]
        rows.append(
            {
                "candidate": candidate,
                "name": CANDIDATE_NAMES[candidate],
                "points": total["points"],
                "judgments": judgments_count,
                "score": total["points"] / judgments_count if judgments_count else None,
            }
        )
    return sorted(
        rows, key=lambda row: row["score"] if row["score"] is not None else -1, reverse=True
    )


def _score_table(title: str, rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        score = f"{row['score'] * 100:.1f}%" if row["score"] is not None else "—"
        body.append(
            f"<tr><th scope='row'>{html.escape(row['name'])}</th><td>{score}</td>"
            f"<td>{row['points']:g}</td><td>{row['judgments']}</td></tr>"
        )
    return (
        f"<section><h2>{html.escape(title)}</h2><table><thead><tr><th>Candidate</th>"
        "<th>Preference score</th><th>Points</th><th>Judgments</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></section>"
    )


REPORT_STYLE = """
:root{color-scheme:dark;font:16px/1.5 system-ui,sans-serif;background:#111;color:#eee}
body{max-width:1500px;margin:auto;padding:1.5rem}a{color:#9ed3c0}h1,h2,h3{line-height:1.2}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
section{margin:1.5rem 0}table{border-collapse:collapse;width:100%}th,td{padding:.5rem;border-bottom:1px solid #555;text-align:left}
.case{border-top:1px solid #555;padding-top:1rem}.context,.outputs{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}
figure{margin:0;min-width:0}img{width:100%;height:340px;object-fit:contain;background:#080808;border:1px solid #444}
figcaption{color:#bbb}.meta{font-size:.9rem;color:#bbb}.artifacts{columns:2}
@media(max-width:700px){body{padding:.75rem}img{height:260px}}
"""


def generate_report(run_path: Path, session_paths: list[Path]) -> Path:
    run_dir = run_path if run_path.is_dir() else run_path.parent
    run = load_run(run_dir)
    prepared_path, prepared = _prepared_for_run(run_dir, run)
    prepared_root = prepared_path.parent
    sessions = []
    session_ids = set()
    all_judgments = []
    for path in session_paths:
        session = _load_session(path)
        session_id = session.get("session_id")
        if not isinstance(session_id, str):
            raise BenchmarkReviewError("review session id must be a UUID")
        if session_id in session_ids:
            raise BenchmarkReviewError(f"duplicate review session {session_id}")
        session_ids.add(session_id)
        all_judgments.extend(validate_session(run, session))
        sessions.append(session)
    if not sessions:
        raise BenchmarkReviewError("at least one completed review session is required")
    case_by_id = {case["id"]: case for case in run["cases"]}
    candidates = [candidate["id"] for candidate in run["candidates"]]
    overall = _score_rows(candidates, all_judgments, lambda _judgment: True)
    paired = _score_rows(
        candidates,
        all_judgments,
        lambda judgment: case_by_id[judgment["case_id"]]["track"] == "paired",
    )
    authentic = _score_rows(
        candidates,
        all_judgments,
        lambda judgment: case_by_id[judgment["case_id"]]["track"] == "authentic",
    )
    all_tags = sorted({tag for case in run["cases"] for tag in case["tags"]})
    tag_tables = "".join(
        _score_table(
            f"Tag: {tag}",
            _score_rows(
                candidates,
                all_judgments,
                lambda judgment, selected=tag: selected in case_by_id[judgment["case_id"]]["tags"],
            ),
        )
        for tag in all_tags
    )
    artifact_counts: dict[str, dict[str, int]] = {
        candidate: {tag: 0 for tag in ARTIFACT_TAGS} for candidate in candidates
    }
    for judgment in all_judgments:
        for side in ("left", "right"):
            candidate = judgment[side]["candidate_id"]
            for tag in judgment["artifacts"][side]:
                artifact_counts[candidate][tag] += 1
    artifact_rows = "".join(
        f"<tr><th scope='row'>{html.escape(CANDIDATE_NAMES[candidate])}</th>"
        + "".join(f"<td>{artifact_counts[candidate][tag]}</td>" for tag in ARTIFACT_TAGS)
        + "</tr>"
        for candidate in candidates
    )
    artifacts = (
        "<section><h2>Reviewer-tagged artifacts</h2><table><thead><tr><th>Candidate</th>"
        + "".join(f"<th>{html.escape(tag)}</th>" for tag in ARTIFACT_TAGS)
        + f"</tr></thead><tbody>{artifact_rows}</tbody></table></section>"
    )
    outputs = {(item["case_id"], item["candidate_id"]): item for item in run["outputs"]}
    prepared_by_case = {case["id"]: case for case in prepared["cases"]}
    gallery = []
    report_path = run_dir / "report.html"
    for case in run["cases"]:
        prepared_case = prepared_by_case[case["id"]]
        source = prepared_root / prepared_case["input"]
        figures = [
            f"<figure><img src='{html.escape(_relative_uri(source, run_dir))}' alt='Source for "
            f"{html.escape(case['title'])}'><figcaption>Source</figcaption></figure>"
        ]
        if prepared_case["reference"]:
            reference = prepared_root / prepared_case["reference"]
            figures.append(
                f"<figure><img src='{html.escape(_relative_uri(reference, run_dir))}' "
                f"alt='Reference for {html.escape(case['title'])}'><figcaption>Reference</figcaption></figure>"
            )
        output_figures = []
        for candidate in candidates:
            item = outputs[(case["id"], candidate)]
            diagnostic = item["diagnostics"]
            diagnostic_text = (
                f" · MAE {diagnostic['mae']:.2f} · PSNR {diagnostic['psnr']:.2f} dB"
                if diagnostic
                else ""
            )
            output_figures.append(
                f"<figure><img src='{html.escape(item['path'])}' alt='"
                f"{html.escape(CANDIDATE_NAMES[candidate])} output for {html.escape(case['title'])}'>"
                f"<figcaption>{html.escape(CANDIDATE_NAMES[candidate])}{diagnostic_text}<br>"
                f"{item['elapsed_seconds']:.2f}s · {item['tile_size']}px tile · "
                f"{html.escape(item['precision'])}</figcaption></figure>"
            )
        tags = ", ".join(case["tags"])
        gallery.append(
            f"<article class='case'><h3>{html.escape(case['title'])}</h3><p class='meta'>"
            f"{html.escape(case['track'])} · {html.escape(tags)} · "
            f"<a href='{html.escape(prepared_case['source_page'])}'>public-domain source</a></p>"
            f"<div class='context'>{''.join(figures)}</div><div class='outputs'>{''.join(output_figures)}</div></article>"
        )
    candidate_meta = "".join(
        f"<li><strong>{html.escape(item['name'])}</strong>: {html.escape(item['adapter_name'])}, "
        f"{html.escape(item['device'])}, {html.escape(item['precision'])}</li>"
        for item in run["candidates"]
    )
    body = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>Upscaler benchmark report</title>"
        f"<style>{REPORT_STYLE}</style></head><body><h1>Perceptual-quality benchmark</h1>"
        f"<p>{len(sessions)} local review session(s), {len(all_judgments)} pairwise judgments. "
        "A win is 1 point and a tie is 0.5; cannot-judge responses are excluded. Pixel metrics "
        "are diagnostics only. These counts do not establish statistical significance.</p>"
        f"<ul>{candidate_meta}</ul><div class='summary'>{_score_table('Overall', overall)}"
        f"{_score_table('Paired-reference track', paired)}"
        f"{_score_table('Authentic-degradation track', authentic)}</div>{artifacts}"
        f"<details><summary>Scores by content/degradation tag</summary>{tag_tables}</details>"
        f"<section><h2>Real outputs</h2>{''.join(gallery)}</section></body></html>"
    )
    report_path.write_text(body, encoding="utf-8")
    aggregate = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "generated_at": _utc_now(),
        "session_ids": sorted(session_ids),
        "review_sessions": len(sessions),
        "judgments": len(all_judgments),
        "scores": {"overall": overall, "paired": paired, "authentic": authentic},
        "artifact_counts": artifact_counts,
    }
    aggregate_path = run_dir / "report.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path
