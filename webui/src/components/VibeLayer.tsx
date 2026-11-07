import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { VibeSettings } from "../vibe";

/** Floating background emojis for the non-Professional vibes. Fixed behind
 * the layout, never takes pointer events, hidden for reduced motion. */

interface Item {
  key: string;
  kind: "heart" | "plus" | "rocket" | "ufo";
  glyph: string;
  style: CSSProperties;
}

const rand = (a: number, b: number) => a + Math.random() * (b - a);
const randInt = (a: number, b: number) => Math.floor(rand(a, b + 1));
const pick = <T,>(xs: T[]) => xs[Math.floor(Math.random() * xs.length)];

function shuffle<T>(xs: T[]): T[] {
  for (let i = xs.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [xs[i], xs[j]] = [xs[j], xs[i]];
  }
  return xs;
}

const vars = (v: Record<string, string | number>) => v as CSSProperties;

function floater(i: number, kind: Item["kind"], glyph: string, extra: CSSProperties = {}): Item {
  return {
    key: `${kind}-${i}`,
    kind,
    glyph,
    style: {
      left: `${randInt(2, 96)}%`,
      ...vars({
        "--emoji-size": `${randInt(20, 40)}px`,
        "--duration": `${rand(16, 30).toFixed(1)}s`,
        "--delay": `${rand(0, 3).toFixed(1)}s`,
      }),
      ...extra,
    },
  };
}

function positivity(nHearts: number, nPluses: number): Item[] {
  const hearts = ["💜", "❤️", "💙"];
  const plusColors = ["#00BCD4", "#FFD600", "#E040FB"];
  const items: Item[] = [];
  for (let i = 0; i < nHearts; i++)
    items.push(
      floater(i, "heart", pick(hearts), vars({
        "--rotation": `${pick([-360, -180, 180, 360])}deg`,
        "--peak-opacity": rand(0.1, 0.2).toFixed(2),
      })),
    );
  for (let i = 0; i < nPluses; i++)
    items.push(
      floater(i, "plus", "✚", {
        ...vars({ "--rotation": "0deg", "--peak-opacity": rand(0.12, 0.22).toFixed(2) }),
        color: pick(plusColors),
        fontWeight: 900,
      }),
    );
  return shuffle(items);
}

function space(nRockets: number, nUfos: number): Item[] {
  const items: Item[] = [];
  for (let i = 0; i < nRockets; i++)
    items.push(floater(i, "rocket", "🚀", vars({ "--peak-opacity": rand(0.1, 0.2).toFixed(2) })));
  for (let i = 0; i < nUfos; i++)
    items.push({
      key: `ufo-${i}`,
      kind: "ufo",
      glyph: "🛸",
      style: {
        top: `${randInt(10, 80)}%`,
        left: 0,
        ...vars({
          "--emoji-size": `${randInt(24, 36)}px`,
          "--duration": `${rand(35, 55).toFixed(1)}s`,
          "--delay": `${(i === 0 ? rand(0, 5) : rand(15, 45)).toFixed(1)}s`,
          "--peak-opacity": rand(0.15, 0.25).toFixed(2),
        }),
      },
    });
  return items;
}

const overlap = (a: DOMRect, b: DOMRect) =>
  !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);

export default function VibeLayer({ settings }: { settings: VibeSettings }) {
  const { vibe, hearts, pluses, rockets, ufos } = settings;
  const ref = useRef<HTMLDivElement>(null);
  const [booms, setBooms] = useState<{ id: number; x: number; y: number }[]>([]);

  const items = useMemo(() => {
    if (vibe === "Positivity boost") return positivity(hearts, pluses);
    if (vibe === "Space nerd") return space(rockets, ufos);
    return [];
  }, [vibe, hearts, pluses, rockets, ufos]);

  // Rocket meets UFO: 💥, and one or both of them are gone.
  useEffect(() => {
    if (vibe !== "Space nerd" || !rockets || !ufos) return;
    const cooldown = new WeakMap<Element, number>();
    let nextId = 0;
    const timer = window.setInterval(() => {
      const root = ref.current;
      if (!root) return;
      const now = Date.now();
      const rs = root.querySelectorAll<HTMLElement>(".rocket");
      const us = root.querySelectorAll<HTMLElement>(".ufo");
      rs.forEach((r) => {
        if ((cooldown.get(r) ?? 0) > now || r.style.display === "none") return;
        const rr = r.getBoundingClientRect();
        if (!rr.width) return;
        for (const u of us) {
          if ((cooldown.get(u) ?? 0) > now || u.style.display === "none") continue;
          const ur = u.getBoundingClientRect();
          if (!ur.width || !overlap(rr, ur)) continue;
          const id = nextId++;
          setBooms((b) => [
            ...b,
            { id, x: (rr.left + rr.right + ur.left + ur.right) / 4, y: (rr.top + rr.bottom + ur.top + ur.bottom) / 4 },
          ]);
          window.setTimeout(() => setBooms((b) => b.filter((x) => x.id !== id)), 900);
          const fate = Math.floor(Math.random() * 3);
          if (fate !== 1) u.style.display = "none";
          if (fate !== 0) r.style.display = "none";
          cooldown.set(r, now + 1000);
          cooldown.set(u, now + 1000);
          return;
        }
      });
    }, 150);
    return () => window.clearInterval(timer);
  }, [vibe, rockets, ufos, items]);

  if (!items.length) return null;
  return (
    <div className="vibe-layer" ref={ref} aria-hidden="true">
      {items.map((it) => (
        <span key={it.key} className={it.kind} style={it.style}>
          {it.glyph}
        </span>
      ))}
      {booms.map((b) => (
        <span key={`boom-${b.id}`} className="boom" style={{ left: b.x, top: b.y }}>
          💥
        </span>
      ))}
    </div>
  );
}
