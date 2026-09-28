import { api, type SubjectBlock } from "../api";
import { resolveSessionPath } from "../filelink";
import { useUIActions } from "../UIContext";
import { MarkdownBody } from "./MarkdownBody";
import { VOCAB } from "../vocabulary";

/** A path in a block's prose (a candidate's full plan, the plan report)
 * opens in the Files tab, as it does from a chat message. */
function useFileClick(sessionId: string) {
  const { openInFiles } = useUIActions();
  return async (token: string) => {
    const resolved = await resolveSessionPath(sessionId, token);
    openInFiles(resolved ?? "");
  };
}

/** "Open full plan" beside a candidate's name, "Open full report" beside a
 * title: the report is a file in the session, shown in the Files tab. */
export function OpenReport({ path, label }: { path?: string | null; label: string }) {
  const { openInFiles } = useUIActions();
  if (!path) return null;
  return (
    <button
      type="button"
      className="qs-open-report"
      onClick={(e) => {
        e.stopPropagation();
        openInFiles(path);
      }}
    >
      {label} ↗
    </button>
  );
}

/** Renders a question's subject — what is under review, as blocks — the
 * React twin of the shell's `Widgets._render_blocks`. One renderer per
 * block type of `scilink.hitl.SUBJECT_BLOCKS`. A labeled block is a
 * section shaped like the console printout: its label (with the emoji) on
 * a heading line, the whole content below it at full width — a reviewer
 * reads all of it before deciding, so nothing is clamped or hidden. A
 * candidates block becomes selectable cards when the panel passes
 * `choice` / `onChoose`. */
export function SubjectBlocks({
  sessionId,
  blocks,
  choice,
  onChoose,
}: {
  sessionId: string;
  blocks: SubjectBlock[];
  choice?: number | null;
  onChoose?: (idx: number) => void;
}) {
  return (
    <div className="qs-sections">
      {blocks.map((b, i) => (
        <Section key={i} label={b.type === "notice" ? undefined : b.label}>
          <Block sessionId={sessionId} block={b} choice={choice} onChoose={onChoose} />
        </Section>
      ))}
    </div>
  );
}

function Section({ label, children }: { label?: string; children: React.ReactNode }) {
  return (
    <div className="qs-section">
      {label && <div className="qs-section-label">{label}</div>}
      <div className="qs-section-body">{children}</div>
    </div>
  );
}

function Block({
  sessionId,
  block,
  choice,
  onChoose,
}: {
  sessionId: string;
  block: SubjectBlock;
  choice?: number | null;
  onChoose?: (idx: number) => void;
}) {
  const onFileClick = useFileClick(sessionId);
  switch (block.type) {
    case "text":
      return (
        <div className="qs-text">
          {/* an agent's prose is full of "~303 cm^-1": never strikethrough */}
          <MarkdownBody text={block.markdown} escapeTilde onFileClick={onFileClick} />
        </div>
      );
    case "fields":
      return (
        <dl className="qs-fields">
          {block.items.map((f, i) => (
            <div key={i} className="qs-field">
              <dt>{f.label}</dt>
              <dd className={f.flag ? `flag-${f.flag}` : undefined}>
                {f.value === null || f.value === undefined ? "—" : String(f.value)}
                {f.unit ? ` ${f.unit}` : ""}
              </dd>
            </div>
          ))}
        </dl>
      );
    case "chips":
      return (
        <div className="qs-chips">
          {block.items.map((c, i) => (
            <span key={i} className="qs-chip">
              {c}
            </span>
          ))}
        </div>
      );
    case "steps":
      return (
        <ol className="qs-steps">
          {block.items.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ol>
      );
    case "table":
      return (
        <div className="qs-table-wrap">
          <table className="qs-table">
            {block.caption && <caption>{block.caption}</caption>}
            <thead>
              <tr>
                {block.columns.map((c, i) => (
                  <th key={i}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((r, i) => (
                <tr key={i}>
                  {r.map((c, j) => (
                    <td key={j}>{c === null || c === undefined ? "" : String(c)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "figure":
      return (
        <figure className="qs-figure">
          {block.path ? (
            <img
              className="preview"
              src={api.fileUrl(sessionId, block.path)}
              alt={block.caption ?? block.path}
            />
          ) : (
            <span className="caption">figure: {block.file ?? "(not in this session)"}</span>
          )}
          {block.caption && <figcaption className="caption">{block.caption}</figcaption>}
        </figure>
      );
    case "candidates":
      return (
        <div className="qs-candidates">
          {block.items.map((c) => {
            const isPick = c.idx === block.pick;
            const selected = choice === c.idx;
            const selectable = Boolean(onChoose);
            return (
              <div
                key={c.idx}
                className={
                  "qs-candidate" +
                  (selected ? " selected" : "") +
                  (selectable ? " selectable" : "")
                }
                role={selectable ? "radio" : undefined}
                aria-checked={selectable ? selected : undefined}
                tabIndex={selectable ? 0 : undefined}
                onClick={() => onChoose?.(c.idx)}
                onKeyDown={(e) => {
                  if (selectable && (e.key === " " || e.key === "Enter")) {
                    e.preventDefault();
                    onChoose?.(c.idx);
                  }
                }}
              >
                <div className="qs-candidate-head">
                  {selectable && (
                    <input
                      type="radio"
                      name="subject-candidate"
                      checked={selected}
                      onChange={() => onChoose?.(c.idx)}
                    />
                  )}
                  <b>
                    {VOCAB.names.candidate} {c.idx}
                  </b>
                  {c.name && <span className="qs-candidate-name">{c.name}</span>}
                  {c.metric !== undefined && c.value !== undefined && (
                    <span className="qs-chip">
                      {c.metric}={String(c.value)}
                    </span>
                  )}
                  {c.approved !== undefined && (
                    <span className={"qs-chip " + (c.approved ? "flag-ok" : "flag-bad")}>
                      {c.approved ? "✓ approved" : "✗ below gate"}
                    </span>
                  )}
                  {isPick && <span className="qs-chip qs-pick">{VOCAB.names.judge_pick}</span>}
                  <OpenReport path={c.report} label="Open full plan" />
                </div>
                {c.judge_comment && <p className="caption">{c.judge_comment}</p>}
                {c.body && (
                  <div className="qs-candidate-body">
                    <MarkdownBody text={c.body} escapeTilde onFileClick={onFileClick} />
                  </div>
                )}
                {c.figure && (
                  <img
                    className="preview"
                    src={api.fileUrl(sessionId, c.figure)}
                    alt={`${VOCAB.names.candidate} ${c.idx}`}
                  />
                )}
              </div>
            );
          })}
          {block.reasoning && (
            <p className="qs-reasoning">
              <b>Judge.</b> {block.reasoning}
            </p>
          )}
          {block.caveats && block.caveats.length > 0 && (
            <ul className="qs-caveats">
              {block.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          )}
        </div>
      );
    case "compare":
      return (
        <div className="qs-compare">
          {[block.left, block.right].map((side, i) => (
            <div key={i} className="qs-compare-side">
              <div className="qs-compare-label">{side.label}</div>
              <SubjectBlocks sessionId={sessionId} blocks={side.blocks} />
            </div>
          ))}
        </div>
      );
    case "notice":
      return (
        <div className={"feedback-notice" + (block.tone === "warn" ? " warn" : "")}>
          <strong>{block.title}</strong>
          {block.lines.length > 0 && (
            <ul>
              {block.lines.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          )}
        </div>
      );
    default:
      return null;
  }
}
