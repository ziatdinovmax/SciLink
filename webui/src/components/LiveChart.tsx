import { useEffect, useMemo, useRef, useState } from "react";

/** A small SVG line chart for the Live tab: one measured series, an optional
 * overlay in the second series colour (the fitted model over the data), an
 * optional neutral dashed reference (a known value, when there is one), point
 * markers for flagged frames and vertical rules for events. One y-axis, a
 * crosshair that snaps to the nearest x, and a tooltip that lists every series
 * at that x. `compact` is the small-multiple form: several of these stacked,
 * one quantity each, instead of several quantities on one axis. */

export interface ChartPoint {
  x: number;
  y: number | null;
}
export interface ChartMarker {
  x: number;
  label: string;
}

const MARGIN = { top: 12, right: 14, bottom: 30, left: 58 };
const MARGIN_COMPACT = { top: 6, right: 14, bottom: 18, left: 58 };
const MARGIN_COMPACT_LABELLED = { top: 6, right: 14, bottom: 30, left: 58 };

function niceTicks(lo: number, hi: number, n: number): number[] {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step)
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** An uncertainty or a fit statistic: two or three significant figures, not five. */
export function fmtShort(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !isFinite(v)) return "";
  return String(parseFloat(v.toPrecision(digits)));
}

export function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
  return String(parseFloat(v.toPrecision(5)));
}

function useWidth(): [React.RefObject<HTMLDivElement>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function LiveChart({
  series,
  seriesLabel,
  overlay,
  overlayLabel,
  reference,
  referenceLabel,
  legend = true,
  compact = false,
  flagged = [],
  rules = [],
  xLabel,
  yLabel,
  xInteger = false,
  height,
}: {
  series: ChartPoint[];
  seriesLabel: string;
  /** A second solid line over the series, e.g. the fitted model over the data. */
  overlay?: ChartPoint[];
  overlayLabel?: string;
  reference?: ChartPoint[];
  referenceLabel?: string;
  /** Small multiples share one legend above the stack. */
  legend?: boolean;
  compact?: boolean;
  /** x positions of flagged frames (drawn as status markers on the series). */
  flagged?: ChartMarker[];
  /** vertical event rules (a re-anchor, a scripted change). */
  rules?: ChartMarker[];
  xLabel?: string;
  yLabel?: string;
  xInteger?: boolean;
  height?: number;
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const M = compact ? (xLabel ? MARGIN_COMPACT_LABELLED : MARGIN_COMPACT) : MARGIN;
  const H = height ?? (compact ? 104 : 220);

  const geo = useMemo(() => {
    const pts = series.filter((p) => p.y !== null && isFinite(p.y as number));
    const refPts = (reference ?? []).filter((p) => p.y !== null && isFinite(p.y as number));
    const ovPts = (overlay ?? []).filter((p) => p.y !== null && isFinite(p.y as number));
    if (!pts.length || width < 120) return null;
    const xs = pts.map((p) => p.x);
    const ys = [...pts, ...refPts, ...ovPts].map((p) => p.y as number);
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = (y1 - y0 || Math.abs(y1) || 1) * 0.08;
    y0 -= pad; y1 += pad;
    const iw = width - M.left - M.right, ih = H - M.top - M.bottom;
    const sx = (v: number) => M.left + ((v - x0) / (x1 - x0)) * iw;
    const sy = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * ih;
    const path = (p: ChartPoint[]) =>
      p.map((q, i) => `${i ? "L" : "M"}${sx(q.x).toFixed(1)},${sy(q.y as number).toFixed(1)}`).join("");
    // A line joins points in the order they came. That is right for a curve
    // sampled along x and wrong for anything else (a sweep that doubles back,
    // unordered samples), where it draws strokes across the plot: those are
    // shown as points instead.
    const dx = xs.slice(1).map((v, i) => v - xs[i]);
    const ordered = dx.every((d) => d >= 0) || dx.every((d) => d <= 0);
    let xt = niceTicks(x0, x1, Math.max(2, Math.floor(iw / 90)));
    if (xInteger) xt = xt.filter((v) => Number.isInteger(v));
    return { pts, refPts, ovPts, sx, sy, path, x0, x1, iw, ih, xt, ordered, yt: niceTicks(y0, y1, compact ? 3 : 4) };
  }, [series, reference, overlay, width, H, M, compact, xInteger]);

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    if (!geo) return;
    const box = e.currentTarget.getBoundingClientRect();
    const xv = geo.x0 + ((e.clientX - box.left) / box.width) * (geo.x1 - geo.x0);
    let best = 0;
    geo.pts.forEach((p, i) => {
      if (Math.abs(p.x - xv) < Math.abs(geo.pts[best].x - xv)) best = i;
    });
    setHover(best);
  };

  const hp = geo && hover !== null ? geo.pts[Math.min(hover, geo.pts.length - 1)] : null;
  const hRef = hp && geo ? geo.refPts.find((p) => p.x === hp.x) : undefined;
  const hOv = hp && geo ? geo.ovPts.find((p) => p.x === hp.x) : undefined;
  const hFlag = hp ? flagged.find((f) => f.x === hp.x) : undefined;
  const dense = (geo?.pts.length ?? 0) > 120;

  return (
    <div className="live-chart" ref={ref}>
      {legend && ((reference && reference.length > 0) || (overlay && overlay.length > 0)) && (
        <div className="live-legend">
          <span><i className="key series" /> {seriesLabel}</span>
          {overlay && overlay.length > 0 && <span><i className="key overlay" /> {overlayLabel}</span>}
          {reference && reference.length > 0 && <span><i className="key reference" /> {referenceLabel}</span>}
          {flagged.length > 0 && <span><i className="key flag">▲</i> flagged</span>}
        </div>
      )}
      {!geo ? (
        <div className="live-chart-empty caption" style={{ height: H }}>No data yet</div>
      ) : (
        <svg width={width} height={H} role="img" aria-label={`${seriesLabel} chart`}>
          {geo.yt.map((v) => (
            <g key={`y${v}`}>
              <line className="grid" x1={M.left} x2={width - M.right} y1={geo.sy(v)} y2={geo.sy(v)} />
              <text className="tick" x={M.left - 6} y={geo.sy(v)} dy="0.32em" textAnchor="end">{fmt(v)}</text>
            </g>
          ))}
          {geo.xt.map((v) => (
            <text key={`x${v}`} className="tick" x={geo.sx(v)} y={H - M.bottom + 14} textAnchor="middle">
              {fmt(v)}
            </text>
          ))}
          <line className="axis" x1={M.left} x2={width - M.right} y1={H - M.bottom} y2={H - M.bottom} />
          {xLabel && (
            <text className="axis-label" x={M.left + geo.iw / 2} y={H - 3} textAnchor="middle">{xLabel}</text>
          )}
          {yLabel && (
            <text className="axis-label" transform={`translate(11,${M.top + geo.ih / 2}) rotate(-90)`} textAnchor="middle">
              {yLabel}
            </text>
          )}
          {rules.filter((r) => r.x >= geo.x0 && r.x <= geo.x1).map((r, i) => (
            <g key={`r${i}`}>
              <line className="rule" x1={geo.sx(r.x)} x2={geo.sx(r.x)} y1={M.top} y2={H - M.bottom} />
              <text
                className="rule-label" y={M.top + 9 + (i % 2) * 11}
                x={geo.sx(r.x) + (geo.sx(r.x) > M.left + geo.iw / 2 ? -4 : 4)}
                textAnchor={geo.sx(r.x) > M.left + geo.iw / 2 ? "end" : "start"}
              >
                {r.label}
              </text>
            </g>
          ))}
          {geo.refPts.length > 1 && <path className="reference" d={geo.path(geo.refPts)} />}
          {geo.ordered ? (
            <>
              <path className={geo.ovPts.length > 1 ? "series thin" : "series"} d={geo.path(geo.pts)} />
              {geo.ovPts.length > 1 && <path className="overlay" d={geo.path(geo.ovPts)} />}
            </>
          ) : (
            <>
              {geo.pts.map((p, i) => (
                <circle key={`s${i}`} className="scatter" cx={geo.sx(p.x)} cy={geo.sy(p.y as number)} r={1.8} />
              ))}
              {geo.ovPts.map((p, i) => (
                <circle key={`o${i}`} className="scatter overlay-pt" cx={geo.sx(p.x)} cy={geo.sy(p.y as number)} r={1.4} />
              ))}
            </>
          )}
          {geo.ordered && !dense && geo.pts.length <= 60 && geo.pts.map((p) => (
            <circle key={p.x} className="dot" cx={geo.sx(p.x)} cy={geo.sy(p.y as number)} r={2.5} />
          ))}
          {flagged.map((f) => {
            const p = geo.pts.find((q) => q.x === f.x);
            if (!p) return null;
            const cx = geo.sx(p.x), cy = geo.sy(p.y as number);
            return <path key={`f${f.x}`} className="flag" d={`M${cx},${cy - 6}L${cx + 5.5},${cy + 4}L${cx - 5.5},${cy + 4}Z`} />;
          })}
          {hp && (
            <g>
              <line className="crosshair" x1={geo.sx(hp.x)} x2={geo.sx(hp.x)} y1={M.top} y2={H - M.bottom} />
              <circle className="hover-dot" cx={geo.sx(hp.x)} cy={geo.sy(hp.y as number)} r={4} />
            </g>
          )}
          <rect
            x={M.left} y={M.top} width={geo.iw} height={geo.ih} fill="transparent"
            onPointerMove={onMove} onPointerLeave={() => setHover(null)}
          />
        </svg>
      )}
      {hp && geo && (
        <div
          className="live-tooltip"
          style={geo.sx(hp.x) > width / 2
            ? { right: width - geo.sx(hp.x) + 10, top: M.top }
            : { left: geo.sx(hp.x) + 10, top: M.top }}
        >
          <div className="tt-head">{xLabel ? `${xLabel} ` : ""}{fmt(hp.x)}</div>
          <div><i className="key series" /> <b>{fmt(hp.y)}</b> <span>{seriesLabel}</span></div>
          {hOv && <div><i className="key overlay" /> <b>{fmt(hOv.y)}</b> <span>{overlayLabel}</span></div>}
          {hRef && <div><i className="key reference" /> <b>{fmt(hRef.y)}</b> <span>{referenceLabel}</span></div>}
          {hFlag && <div className="tt-flag">▲ {hFlag.label}</div>}
        </div>
      )}
    </div>
  );
}
