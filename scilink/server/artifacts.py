"""Per-session artifact discovery — filesystem sweeps, ported from
scilink/ui/app.py (_find_new_images :160, _find_new_html_reports :300,
_find_new_md_documents :355) with the same known-set de-dup semantics
(image identity = path; report identity = path + mtime_ns) and the same
exclusion rules. One ``ArtifactTracker`` per web session replaces
``st.session_state.known_images``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
_UPLOAD_DIRS = {"uploads", "knowledge", "code", "data"}
_MAX_INLINE_BYTES = 60_000
_BULK_STEMS = ("literature_search", "chat_history", "session_log")


def _natural_sort_key(s: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def _img_key(p: Path) -> str:
    """Identity for an image artifact: path + mtime, so a figure rewritten
    in place (a refined ``visualization.png`` at the same path) is seen as
    new and re-surfaced — matching the report/doc sweeps, which have always
    keyed on ``path + mtime_ns``. Falls back to bare path when unstattable."""
    try:
        return f"{p}:{p.stat().st_mtime_ns}"
    except OSError:
        return str(p)


def load_deliverable_titles(session_dir: str) -> Dict[str, str]:
    """{resolved md/html path -> title} from the deliverables manifests."""
    try:
        from scilink.agents.planning_agents.user_interface import load_deliverables
        return {e["path"]: e.get("title") or ""
                for e in load_deliverables(session_dir) if e.get("path")}
    except Exception:
        return {}


class ArtifactTracker:
    """Tracks which session-dir artifacts have already been surfaced."""

    def __init__(self, session_dir: str) -> None:
        self.session_dir = session_dir
        self.known: Set[str] = set()

    # -- resume support ------------------------------------------------
    def mark_all_existing(self) -> None:
        """Pre-mark everything on disk (port of sidebar.py:1218-1233) so the
        first post-resume turn doesn't dump the whole session's artifacts."""
        root = Path(self.session_dir)
        for ext in IMAGE_EXTENSIONS:
            for p in root.rglob(f"*{ext}"):
                self.known.add(_img_key(p))
        for ext in (".html", ".md"):
            for p in root.rglob(f"*{ext}"):
                try:
                    self.known.add(f"{p}:{p.stat().st_mtime_ns}")
                except OSError:
                    self.known.add(str(p))

    def mark_known(self, path: str) -> None:
        self.known.add(_img_key(Path(path)))

    # -- per-turn sweeps ----------------------------------------------
    def find_new_images(self, autonomy: Optional[str] = None) -> List[str]:
        """Port of app.py:160 ``_find_new_images`` (summary_only unused by the
        completion path). ``autonomy`` is the session's autonomy string —
        debug_* sample fits are inlined only in co-pilot, as in Streamlit."""
        session_dir = self.session_dir
        new: List[str] = []
        debug_plots: List[str] = []
        root = Path(session_dir)
        for ext in IMAGE_EXTENSIONS:
            for p in root.rglob(f"*{ext}"):
                if "review" in p.stem:
                    continue
                if p.parent.name == "bo_artifacts" and not p.stem.startswith("step_"):
                    continue
                if _UPLOAD_DIRS & {part for part in p.relative_to(root).parts[:-1]}:
                    continue
                s = str(p)
                key = _img_key(p)
                if key not in self.known:
                    if p.stem.startswith("debug_"):
                        debug_plots.append((key, s))
                    else:
                        self.known.add(key)
                        new.append(s)
        if debug_plots:
            for key, s in debug_plots:
                self.known.add(key)
            if autonomy == "co-pilot":
                paths = [s for _key, s in debug_plots]
                paths.sort(key=_natural_sort_key)
                selected = [paths[0]]
                if len(paths) > 2:
                    selected.append(paths[len(paths) // 2])
                if len(paths) > 1:
                    selected.append(paths[-1])
                new[0:0] = selected
        return new

    def find_new_html_reports(self) -> List[str]:
        """Port of app.py:300 ``_find_new_html_reports``."""
        new: List[str] = []
        for p in Path(self.session_dir).rglob("*.html"):
            s = str(p)
            try:
                key = f"{p}:{p.stat().st_mtime_ns}"
            except OSError:
                key = str(p)
            if key not in self.known:
                self.known.add(key)
                if p.parent.name == "plan_candidates":
                    continue
                if p.stem == "plan_preview":
                    continue
                new.append(s)
        return new

    def find_new_md_documents(self) -> List[str]:
        """Port of app.py:355 ``_find_new_md_documents``."""
        marked: Set[str] = set()
        try:
            from scilink.agents.planning_agents.user_interface import load_deliverables
            marked = {e["path"] for e in load_deliverables(self.session_dir)
                      if e.get("deliverable")}
        except Exception:
            pass
        new: List[str] = []
        for p in Path(self.session_dir).rglob("*.md"):
            s = str(p)
            try:
                key = f"{p}:{p.stat().st_mtime_ns}"
            except OSError:
                key = str(p)
            if key in self.known:
                continue
            is_marked = str(p.resolve()) in marked
            if not is_marked:
                if any(p.stem.startswith(b) for b in _BULK_STEMS):
                    continue
                try:
                    if p.stat().st_size > _MAX_INLINE_BYTES:
                        continue
                except OSError:
                    continue
            self.known.add(key)
            new.append(s)
        return new

    def sweep_turn(self, autonomy: Optional[str] = None) -> Dict[str, Any]:
        """One completed-turn sweep. HTML reports suppress raw images
        (app.py:1009) exactly as in Streamlit. Paths are session-relative."""
        images = self.find_new_images(autonomy)
        reports = self.find_new_html_reports()
        docs = self.find_new_md_documents()
        titles = load_deliverable_titles(self.session_dir)

        def rel(p: str) -> str:
            try:
                return str(Path(p).resolve().relative_to(
                    Path(self.session_dir).resolve()))
            except ValueError:
                return p

        def doc_title(p: str) -> str:
            t = titles.get(str(Path(p).resolve()))
            if t:
                return t
            stem = Path(p).stem
            if stem.startswith("white_paper"):
                return "White Paper"
            if stem.startswith("ideation_report"):
                return "Ideation Report"
            return stem.replace("_", " ").title()

        return {
            "images": [] if reports else [rel(p) for p in images],
            "html_reports": [{"path": rel(p), "name": Path(p).name}
                             for p in reports],
            "md_reports": [{"path": rel(p), "name": Path(p).name,
                            "title": doc_title(p)} for p in docs],
        }
