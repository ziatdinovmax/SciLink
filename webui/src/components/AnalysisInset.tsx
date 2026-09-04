/** Floating "watch the analysis" inset: a draggable, dismissible
 * picture-in-picture that shows figures the agent writes as the turn runs.
 * Auto-advances to the newest figure unless the user is browsing back through
 * the filmstrip; a filename + branch label gives context during a fan-out. */

import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useUIActions } from "../UIContext";
import type { LiveImage } from "../App";

export function AnalysisInset({
  sessionId,
  images,
  running,
}: {
  sessionId: string;
  images: LiveImage[];
  running: boolean;
}) {
  const { openInFiles } = useUIActions();
  const [idx, setIdx] = useState(0); // offset from newest: 0 = latest
  const [dismissed, setDismissed] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const drag = useRef<{ dx: number; dy: number } | null>(null);

  const n = images.length;
  // Auto-follow the newest figure: whenever a new one arrives, snap to it
  // unless the user has paged back (idx > 0 is preserved by the effect below).
  useEffect(() => {
    if (n > 0) setIdx((cur) => (cur === 0 ? 0 : cur + 1)); // keep same image
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [n]);

  // A new turn (images reset to empty) re-shows a dismissed inset and drops
  // any browsed-back offset, so the next turn follows its latest figure.
  useEffect(() => {
    if (n === 0) {
      setDismissed(false);
      setIdx(0);
    }
  }, [n]);

  if (n === 0 || dismissed) return null;

  const clamped = Math.min(idx, n - 1);
  const img = images[n - 1 - clamped];
  const isLatest = clamped === 0;

  const onDown = (e: React.PointerEvent) => {
    const box = (e.currentTarget.closest(".analysis-inset") as HTMLElement)
      .getBoundingClientRect();
    drag.current = { dx: e.clientX - box.left, dy: e.clientY - box.top };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    setPos({
      x: Math.max(4, e.clientX - drag.current.dx),
      y: Math.max(4, e.clientY - drag.current.dy),
    });
  };
  const onUp = () => (drag.current = null);

  const style = pos
    ? { left: pos.x, top: pos.y, right: "auto", bottom: "auto" }
    : undefined;

  return (
    <div className="analysis-inset" style={style} role="dialog" aria-label="Live analysis figure">
      <div
        className="inset-bar"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
      >
        <span className={`inset-dot ${running ? "live" : ""}`} aria-hidden="true" />
        <span className="inset-title" title={img.path}>
          {img.branch ? <em>{img.branch}</em> : null}
          {img.label || "figure"}
        </span>
        <div className="inset-actions">
          <button
            className="inset-btn"
            onClick={() => openInFiles(img.path)}
            title="Open in Files"
          >
            📁
          </button>
          <button
            className="inset-btn"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? "Expand" : "Collapse"}
          >
            {collapsed ? "▢" : "—"}
          </button>
          <button className="inset-btn" onClick={() => setDismissed(true)} title="Hide">
            ✕
          </button>
        </div>
      </div>

      {!collapsed && (
        <>
          <div
            className="inset-figure"
            onClick={() => openInFiles(img.path)}
            title="Open in Files"
          >
            <img
              key={`${img.path}:${img.v ?? ""}`}
              src={api.fileUrl(sessionId, img.path, img.v)}
              alt={img.label}
            />
          </div>
          <div className="inset-foot">
            <button
              className="inset-btn"
              disabled={clamped >= n - 1}
              onClick={() => setIdx(clamped + 1)}
              title="Older figure"
            >
              ‹
            </button>
            <span className="inset-count">
              {isLatest ? (running ? "latest · live" : "latest") : `−${clamped}`} ·{" "}
              {n}
            </span>
            <button
              className="inset-btn"
              disabled={clamped <= 0}
              onClick={() => setIdx(clamped - 1)}
              title="Newer figure"
            >
              ›
            </button>
          </div>
        </>
      )}
    </div>
  );
}
