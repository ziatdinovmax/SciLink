import { useCallback, useEffect, useState, type ReactNode } from "react";
import { api, type InstrumentMemory as Memory, type RememberedInstrument } from "../api";

function day(stamp?: string | null): string {
  return stamp ? stamp.replace("T", " ").slice(0, 16) : "never";
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Where a remembered recipe came from, in words. */
function origin(source?: string | null): string {
  if (!source) return "a reference analysis";
  if (source.startsWith("reference")) return "a reference analysis";
  if (source.startsWith("reanchor")) return "a rebuild during a run";
  if (source === "bank") return "the script bank";
  if (source === "anchor") return "a past analysis";
  return source;
}

/** What the instruments on this machine remember: their recipes and past runs.
 * Read from the store when opened. A recipe can be forgotten here, which never
 * touches a run that is using it (a run replays its own copy). */
export function InstrumentMemory({ currentId, refreshKey, info }: { currentId?: string; refreshKey?: string; info?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [known, setKnown] = useState<RememberedInstrument[] | null>(null);
  const [canForget, setCanForget] = useState(false);
  const [shown, setShown] = useState<string | null>(null);
  const [memory, setMemory] = useState<Memory | null>(null);
  const [asking, setAsking] = useState<string | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    api.liveInstruments()
      .then((r) => { setKnown(r.instruments); setCanForget(r.can_forget); setError(""); })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => { load(); }, [load, refreshKey]);
  useEffect(() => {
    if (!open || !shown) { setMemory(null); return; }
    api.liveInstrument(shown).then(setMemory).catch((e) => setError(String(e.message ?? e)));
  }, [open, shown, refreshKey]);
  useEffect(() => {
    if (open && !shown && known?.length) {
      setShown((known.find((k) => k.id === currentId) ?? known[0]).key);
    }
  }, [open, shown, known, currentId]);

  const forget = (recipeId: string) => {
    if (!shown) return;
    api.liveForgetRecipe(shown, recipeId)
      .then((m) => { setMemory(m); setAsking(null); load(); })
      .catch((e) => setError(String(e.message ?? e)));
  };

  if (!known?.length) return null;
  return (
    <details className="live-options live-memory" open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Instrument memory ({known.length}) {info}</summary>
      {error && <p className="caption live-memory-error">{error}</p>}
      <div className="live-memory-list">
        {known.map((k) => (
          <button key={k.key} type="button" className={k.key === shown ? "on" : ""} onClick={() => setShown(k.key)}>
            <strong>{k.id}</strong>
            {k.id === currentId && <span className="live-memory-tag">this instrument</span>}
            <span className="caption">
              {[k.technique, plural(k.recipes, "recipe"), plural(k.runs, "run"), `last seen ${day(k.last_seen)}`]
                .filter(Boolean).join(" · ")}
            </span>
          </button>
        ))}
      </div>
      {memory && memory.instrument.key === shown && (
        <div className="live-memory-detail">
          <h4>Recipes</h4>
          {memory.recipes.length === 0 && <p className="caption">None kept. The next run analyses its reference.</p>}
          {memory.recipes.length > 0 && (
            <div className="live-memory-scroll">
              <table>
                <thead>
                  <tr><th>Tracks</th><th>Sample</th><th>Recalled</th><th>Last used</th><th>From</th><th>Size</th><th /></tr>
                </thead>
                <tbody>
                  {memory.recipes.map((r) => {
                    const names = Object.keys(r.outputs).length ? Object.keys(r.outputs) : r.reports.slice(0, 4);
                    return (
                      <tr key={r.recipe_id}>
                        <td title={Object.entries(r.outputs).map(([n, d]) => `${n}: ${d}`).join("\n")}>
                          {names.join(", ") || "not recorded"}
                        </td>
                        <td>{r.sample || "not recorded"}</td>
                        <td>{r.uses ? plural(r.uses, "time") : "never"}</td>
                        <td>{day(r.last_used)}</td>
                        <td>{origin(r.source)}</td>
                        <td>{r.size_mb} MB</td>
                        <td className="live-memory-act">
                          {canForget && asking !== r.recipe_id && (
                            <button type="button" onClick={() => setAsking(r.recipe_id)}>Forget</button>
                          )}
                          {canForget && asking === r.recipe_id && (
                            <>
                              <button type="button" className="danger" onClick={() => forget(r.recipe_id)}>Forget it</button>
                              <button type="button" onClick={() => setAsking(null)}>Keep</button>
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <h4>Runs</h4>
          {memory.runs.length === 0 && <p className="caption">No finished run yet.</p>}
          {memory.runs.length > 0 && (
            <div className="live-memory-scroll">
              <table>
                <thead><tr><th>When</th><th>Frames</th><th>Clean</th><th>Changes</th><th>Rebuilds</th><th>Audits</th></tr></thead>
                <tbody>
                  {memory.runs.map((run, i) => (
                    <tr key={i}>
                      <td>{day(run.when)}</td>
                      <td>{run.frames}</td>
                      <td>{run.clean_frames}</td>
                      <td title={run.novelties.map((n) => `frame ${n.step ?? "?"}${n.region ? `, ${n.region}` : ""}${n.onset === "gradual" ? ", gradual" : ""}`).join("\n")}>
                        {run.novelties.length}
                      </td>
                      <td>{run.reanchors ?? 0}</td>
                      <td>{run.audits ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </details>
  );
}
