import { useEffect, useRef, useState } from "react";
import { api, type StructureModel } from "../api";

// An interactive 3D view of an atomic structure (3Dmol.js, loaded on first
// use so it stays out of the main bundle). The server reads the file with
// ASE (previews.structure_model) and turns a slab surface-up; the colours
// are ASE's, as in the drawn views.

type Look = "ball" | "space";

// Rotation the view opens on: a slab is shown from the side with its top
// tilted toward the viewer, anything else slightly off-axis so depth reads.
const OPEN_TILT = { slab: [40, 20], other: [20, 30] } as const;

function panelBackground(): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--bg-panel").trim();
  return v || "#ffffff";
}

function toXyz(m: StructureModel): string {
  const lines = m.symbols.map((s, i) => `${s} ${m.positions[i].join(" ")}`);
  return `${m.symbols.length}\n${m.title}\n${lines.join("\n")}\n`;
}

// The cell's twelve edges, from its three vectors at the origin.
function cellEdges(cell: number[][]): [number[], number[]][] {
  const [a, b, c] = cell;
  const add = (...vs: number[][]) => [0, 1, 2].map((k) => vs.reduce((s, v) => s + v[k], 0));
  const o = [0, 0, 0];
  const corners = [o, a, b, c, add(a, b), add(a, c), add(b, c), add(a, b, c)];
  const pairs = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]];
  return pairs.map(([i, j]) => [corners[i], corners[j]]);
}

function applyLook(viewer: any, m: StructureModel, look: Look) {
  for (const [elem, color] of Object.entries(m.colors)) {
    viewer.setStyle(
      { elem },
      look === "ball"
        ? { sphere: { scale: 0.3, color }, stick: { radius: 0.12, color } }
        : { sphere: { scale: 0.9, color } },
    );
  }
}

// A 3Dmol viewer outlives clear(): its spin timer stops only once its
// context is lost, and 3Dmol renders through one shared offscreen context
// that never is, so a spinning viewer left alone renders forever. So it is
// stopped explicitly, and its observers of the container disconnected. (Its
// window resize and body mouseup listeners are bound in 3Dmol's constructor
// and cannot be removed; the inset keeps one viewer per structure folder so
// few are ever made.)
// Where 3Dmol falls back to a WebGL context per canvas (no OffscreenCanvas),
// that context is released too; in the shared mode the canvas holds only a
// bitmap renderer and the release is a no-op.
function disposeViewer(v: any) {
  v.spin(false);
  v.divwatcher?.disconnect();
  v.intwatcher?.disconnect();
  // clear() would render once more, into a canvas React has already
  // detached (a zero-size framebuffer); the removals alone do not render.
  v.removeAllSurfaces();
  v.removeAllModels();
  v.removeAllLabels();
  v.removeAllShapes();
  const canvas: HTMLCanvasElement | undefined = v.getCanvas?.();
  const gl = canvas && (canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  gl?.getExtension("WEBGL_lose_context")?.loseContext();
}

// `compact`: the floating inset's form — no toolbar or legend, no mouse
// (a click there opens the file in Files, where the full viewer is), turning
// slowly so the shape reads. The inset keeps one viewer per structure folder,
// so its `path` moves to the folder's newest file (a POSCAR, then its
// CONTCAR): what is shown stays until the newer file is read. `version`
// re-reads a file rewritten in place; the new structure replaces the old in
// the same viewer, view kept. A read that fails (a file caught mid-write)
// leaves a structure already shown in place.
export function StructureViewer({
  sessionId,
  path,
  version,
  compact = false,
}: {
  sessionId: string;
  path: string;
  version?: number;
  compact?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const viewer = useRef<any>(null);
  const openingView = useRef<number[] | null>(null);
  const shownPath = useRef<string | null>(null);
  const lookRef = useRef<Look>("ball");
  const [ready, setReady] = useState(false);
  // The structure with the path it was read from: after a switch of file,
  // the previous file's structure must not open the new file's view.
  const [loaded, setLoaded] = useState<{ path: string; model: StructureModel } | null>(null);
  const loadedPath = useRef<string | null>(null);
  const model = loaded && (loaded.path === path || compact) ? loaded.model : null;
  // A failed read and a viewer that could not start are separate: a later
  // read clears only its own.
  const [readError, setReadError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const error = startError ?? readError;
  const [look, setLook] = useState<Look>("ball");
  const [spinning, setSpinning] = useState(false);

  useEffect(() => {
    let alive = true; // a slower, older read must not replace a newer one
    api.structure(sessionId, path)
      .then((m) => {
        if (!alive) return;
        loadedPath.current = path;
        setLoaded({ path, model: m });
        setReadError(null);
      })
      .catch((e) => {
        if (!alive) return;
        if (loadedPath.current === path || (compact && loadedPath.current)) return;
        setReadError(String(e?.message ?? e));
      });
    return () => {
      alive = false;
    };
  }, [sessionId, path, version, compact]);

  // One viewer per mounted component, in a mount of its own: a viewer
  // disposed later still observes its container, and must not find the
  // next viewer's canvas there.
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    let cancelled = false;
    const mount = document.createElement("div");
    mount.style.cssText = "position:absolute;inset:0";
    el.appendChild(mount);
    let resize: ResizeObserver | null = null;
    let themeWatch: MutationObserver | null = null;
    import("3dmol")
      .then(($3Dmol) => {
        if (cancelled) return;
        viewer.current = $3Dmol.createViewer(mount, {
          backgroundColor: panelBackground(),
          ...(compact ? { nomouse: true } : {}),
        });
        resize = new ResizeObserver(() => viewer.current?.resize());
        resize.observe(el);
        themeWatch = new MutationObserver(() => {
          viewer.current?.setBackgroundColor(panelBackground(), 1);
          viewer.current?.render();
        });
        themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
        setReady(true);
      })
      .catch((e) => !cancelled && setStartError(`Could not start the 3D viewer: ${e?.message ?? e}`));
    return () => {
      cancelled = true;
      resize?.disconnect();
      themeWatch?.disconnect();
      if (viewer.current) disposeViewer(viewer.current);
      viewer.current = null;
      shownPath.current = null;
      mount.remove();
      setReady(false);
    };
  }, [compact]);

  // Draw the structure. A new file opens on the default view; the same file
  // re-read keeps the view (and the spin) the person or the inset has.
  useEffect(() => {
    const v = viewer.current;
    if (!ready || !v || !model) return;
    v.removeAllModels();
    v.removeAllShapes();
    v.removeAllLabels();
    const atoms = v.addModel(toXyz(model), "xyz");
    applyLook(v, model, lookRef.current);
    if (model.cell) {
      for (const [start, end] of cellEdges(model.cell)) {
        v.addCylinder({
          start: { x: start[0], y: start[1], z: start[2] },
          end: { x: end[0], y: end[1], z: end[2] },
          radius: 0.04, color: "#8a94a6", fromCap: 1, toCap: 1,
        });
      }
    }
    if (!compact) {
      v.setHoverable(
        {},
        true,
        (atom: any, vw: any) => {
          if (!atom.label) {
            atom.label = vw.addLabel(`${atom.elem} ${atom.index + 1}`, {
              position: atom, backgroundColor: "#1e2530", backgroundOpacity: 0.85,
              fontColor: "#ffffff", fontSize: 12, inFront: true,
            });
          }
        },
        (atom: any, vw: any) => {
          if (atom.label) {
            vw.removeLabel(atom.label);
            delete atom.label;
          }
        },
      );
    }
    if (shownPath.current !== path) {
      shownPath.current = path;
      // Frame the atoms, not the cell: a slab's cell is mostly vacuum.
      // (An empty selection would take in the cell's edges too.)
      v.zoomTo({ model: atoms });
      const [x, y] = model.slab ? OPEN_TILT.slab : OPEN_TILT.other;
      v.rotate(x, "x");
      v.rotate(y, "y");
      v.zoom(0.9);
      openingView.current = v.getView();
      if (compact) v.spin("y", 0.5, true); // only while the inset is visible
    }
    v.render();
  }, [ready, model, path, compact]);

  const changeLook = (next: Look) => {
    setLook(next);
    lookRef.current = next;
    if (viewer.current && model) {
      applyLook(viewer.current, model, next);
      viewer.current.render();
    }
  };

  const toggleSpin = () => {
    const next = !spinning;
    setSpinning(next);
    viewer.current?.spin(next ? "y" : false);
  };

  const resetView = () => {
    const v = viewer.current;
    if (!v || !openingView.current) return;
    v.setView(openingView.current);
    v.render();
  };

  // The host stays mounted whatever is shown, so the viewer in it does too.
  const box = (
    <div
      ref={host}
      className={`structure-3d${compact ? " compact" : ""}`}
      style={error ? { display: "none" } : undefined}
    />
  );
  const errorLine = error && (
    <p className="caption" style={{ padding: compact ? "8px 12px" : "8px 0" }}>{error}</p>
  );
  if (compact) return <>{errorLine}{box}</>;
  return (
    <div>
      {!error && (
        <div className="preview-toolbar">
          <button className={look === "ball" ? "primary" : ""} onClick={() => changeLook("ball")}>
            Ball and stick
          </button>
          <button className={look === "space" ? "primary" : ""} onClick={() => changeLook("space")}>
            Space-filling
          </button>
          <button className={spinning ? "primary" : ""} onClick={toggleSpin}>Spin</button>
          <button onClick={resetView}>Reset view</button>
          <span className="caption">drag to rotate · scroll to zoom · hover an atom</span>
        </div>
      )}
      {errorLine}
      {!model && !error && <p className="caption">Loading…</p>}
      {box}
      {model && !error && (
        <div className="structure-legend caption">
          <strong>{model.title}</strong>
          {Object.entries(model.colors).map(([s, c]) => (
            <span key={s} className="structure-chip">
              <span className="structure-swatch" style={{ background: c }} />
              {s}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
