import { api, type SubjectBlock } from "../api";
import { MarkdownBody } from "./MarkdownBody";
import { VOCAB } from "../vocabulary";

/** Renders a question's subject — what is under review, as blocks — the
 * React twin of the shell's `Widgets._render_blocks`. One renderer per
 * block type of `scilink.hitl.SUBJECT_BLOCKS`; a candidates block becomes
 * selectable cards when the panel passes `choice` / `onChoose`. */
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
    <>
      {blocks.map((b, i) => (
        <Block
          key={i}
          sessionId={sessionId}
          block={b}
          choice={choice}
          onChoose={onChoose}
        />
      ))}
    </>
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
  switch (block.type) {
    case "text":
      return (
        <div className="sb-text">
          <MarkdownBody text={block.markdown} />
        </div>
      );
    case "fields":
      return (
        <dl className="sb-fields">
          {block.items.map((f, i) => (
            <div key={i} className="sb-field">
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
        <div className="sb-chips">
          {block.label && <span className="sb-label">{block.label}</span>}
          {block.items.map((c, i) => (
            <span key={i} className="sb-chip">
              {c}
            </span>
          ))}
        </div>
      );
    case "steps":
      return (
        <div className="sb-steps">
          {block.label && <span className="sb-label">{block.label}</span>}
          <ol>
            {block.items.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </div>
      );
    case "table":
      return (
        <div className="sb-table-wrap">
          <table className="sb-table">
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
        <figure className="sb-figure">
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
    case "claims":
      return (
        <ol className="sb-claims">
          {block.items.map((c, i) => (
            <li key={i}>
              <details>
                <summary>{c.claim}</summary>
                <div className="sb-claim-body">
                  {c.impact && (
                    <p>
                      <b>Impact.</b> {c.impact}
                    </p>
                  )}
                  {c.question && (
                    <p>
                      <b>Question.</b> {c.question}
                    </p>
                  )}
                  {c.keywords && c.keywords.length > 0 && (
                    <div className="sb-chips">
                      {c.keywords.map((k, j) => (
                        <span key={j} className="sb-chip">
                          {k}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </details>
            </li>
          ))}
        </ol>
      );
    case "candidates":
      return (
        <div className="sb-candidates">
          {block.items.map((c) => {
            const isPick = c.idx === block.pick;
            const selected = choice === c.idx;
            const selectable = Boolean(onChoose);
            return (
              <div
                key={c.idx}
                className={
                  "sb-candidate" +
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
                <div className="sb-candidate-head">
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
                  {c.name && <span className="sb-candidate-name">{c.name}</span>}
                  {c.metric !== undefined && c.value !== undefined && (
                    <span className="sb-chip">
                      {c.metric}={String(c.value)}
                    </span>
                  )}
                  {c.approved !== undefined && (
                    <span className={"sb-chip " + (c.approved ? "flag-ok" : "flag-bad")}>
                      {c.approved ? "✓ approved" : "✗ below gate"}
                    </span>
                  )}
                  {isPick && <span className="sb-chip sb-pick">{VOCAB.names.judge_pick}</span>}
                </div>
                {c.judge_comment && <p className="caption">{c.judge_comment}</p>}
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
            <p className="sb-reasoning">
              <b>Judge.</b> {block.reasoning}
            </p>
          )}
          {block.caveats && block.caveats.length > 0 && (
            <ul className="sb-caveats">
              {block.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          )}
        </div>
      );
    case "compare":
      return (
        <div className="sb-compare">
          {[block.left, block.right].map((side, i) => (
            <div key={i} className="sb-compare-side">
              <div className="sb-label">{side.label}</div>
              <SubjectBlocks sessionId={sessionId} blocks={side.blocks} />
            </div>
          ))}
        </div>
      );
    case "notice":
      return (
        <div className={"feedback-notice" + (block.tone === "warn" ? " warn" : "")}>
          <strong>{block.title}</strong>
          <ul>
            {block.lines.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </div>
      );
    default:
      return null;
  }
}
