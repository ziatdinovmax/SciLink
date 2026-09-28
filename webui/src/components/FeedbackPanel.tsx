import { useEffect, useRef, useState } from "react";
import { api, type PresentedQuestion } from "../api";
import { MarkdownBody } from "./MarkdownBody";
import { SubjectBlocks } from "./SubjectBlocks";
import { fill } from "../narration";
import { VOCAB } from "../vocabulary";

/** Renders the parked HITL question: what is under review (the body) and
 * the decision (the row pinned under it). The body is the gate's `subject`
 * blocks when it declared them, else the captured console text and the
 * preview sweep (the legacy path, the React twin of the Streamlit feedback
 * branch). The response contracts are identical on both paths: bare digit /
 * "" for candidate selectors, "y"/"no" for confirm, "keep"/"" for
 * keep-revert, free text or "" elsewhere. */
export function FeedbackPanel({
  sessionId,
  question,
  onRespond,
}: {
  sessionId: string;
  question: PresentedQuestion;
  onRespond: (response: string) => void;
}) {
  const [text, setText] = useState("");
  const [choice, setChoice] = useState<number | null>(
    question.judge_pick ?? null,
  );
  const [sent, setSent] = useState(false);
  // Enter answers a question the way it does at the console prompt, so
  // the question takes the keyboard focus when it appears: the feedback
  // box on the text widgets, the panel itself on the others.
  const focusRef = useRef<HTMLTextAreaElement | HTMLDivElement>(null);
  useEffect(() => {
    focusRef.current?.focus({ preventScroll: true });
  }, [question.request_id]);

  const respond = (response: string) => {
    if (sent) return;
    setSent(true);
    onRespond(response);
  };
  // The empty answer accepts as-is on every surface; the terminal shell
  // shows the same hint next to its prompt.
  const hint = question.labels.accept
    ? fill(VOCAB.enter_accepts_hint, { accept: question.labels.accept })
    : "";
  const isPicker =
    question.widget === "bestofn" || question.widget === "plan_candidates";
  const subject = question.subject;

  // ── the body: what is under review ──────────────────────────
  const previews = question.preview_images.map((p) => {
    const base = p.split("/").pop() ?? p;
    return (
      <figure key={p} style={{ margin: "0 0 8px" }}>
        <img className="preview" src={api.fileUrl(sessionId, p)} alt={base} />
        {question.candidate_captions[base] && (
          <figcaption className="caption">
            {question.candidate_captions[base]}
          </figcaption>
        )}
      </figure>
    );
  });

  const codeFiles = question.code_files.map((f, i) => (
    <details className="card" key={f.name} open={question.code_files.length === 1 && i === 0}>
      <summary>📄 {f.name}</summary>
      <div className="card-body">
        <pre style={{ margin: 0, overflowX: "auto" }}>
          <code>{f.content}</code>
        </pre>
      </div>
    </details>
  ));

  const notice = question.notice ? (
    <div className="feedback-notice">
      <strong>{question.notice.title}</strong>
      <ul>
        {question.notice.lines.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </div>
  ) : null;

  // The captured console text is what a gate WITHOUT a subject shows; with
  // one, the blocks are that text, so it is not shown twice.
  const consoleText =
    question.context_display && !subject ? (
      <div className="context-box">{question.context_display}</div>
    ) : null;

  const fanout = question.widget === "fanout_confirm" ? question.fanout : null;
  // "📋 Proposed fitting plan — single spectrum": the part after the dash
  // is the subtitle.
  const [title, subtitle] = subject
    ? (() => {
        const i = subject.title.indexOf(" — ");
        return i < 0
          ? [subject.title, ""]
          : [subject.title.slice(0, i), subject.title.slice(i + 3)];
      })()
    : ["", ""];
  const body = subject ? (
    <>
      {title && (
        <h4 className="qs-title">
          {title}
          {subtitle && <span className="qs-subtitle">{subtitle}</span>}
        </h4>
      )}
      <SubjectBlocks
        sessionId={sessionId}
        blocks={subject.blocks}
        choice={isPicker ? choice : undefined}
        onChoose={isPicker ? setChoice : undefined}
      />
      {codeFiles}
      {notice}
    </>
  ) : fanout !== null ? (
    <>
      <h4>🔀 Launch parallel multi-dataset analysis?</h4>
      {fanout?.verdict && (
        <MarkdownBody text={`**Complementarity:** ${fanout.verdict}`} />
      )}
      {fanout?.join_axis && <MarkdownBody text={`**Join axis:** ${fanout.join_axis}`} />}
      {fanout && fanout.branches.length > 0 && (
        <MarkdownBody
          text={
            "**Branches** — run concurrently, each seeing the others as auxiliary:\n" +
            fanout.branches.map((b) => `- ${b}`).join("\n")
          }
        />
      )}
      {fanout?.rationale && <MarkdownBody text={`**Why:** ${fanout.rationale}`} />}
      <p className="caption">
        Branches run autonomously — no per-branch approval pauses.
      </p>
    </>
  ) : (
    <>
      {previews}
      {codeFiles}
      {consoleText}
      {notice}
      {isPicker && (
        <>
          <p style={{ marginTop: 0 }}>{question.labels.select}</p>
          <div className="radio-list">
            {(question.candidates ?? []).map((c) => (
              <label key={c.idx}>
                <input
                  type="radio"
                  name="candidate"
                  checked={choice === c.idx}
                  onChange={() => setChoice(c.idx)}
                />
                {c.label}
              </label>
            ))}
          </div>
        </>
      )}
    </>
  );

  // ── the decision ────────────────────────────────────────────
  let decision;
  if (question.widget === "keep_revert") {
    // The primary reply is the gate's own first option ("keep",
    // "consensus", ...); the empty reply is the other.
    const primary = question.options?.[0] || "keep";
    decision = (
      <>
        <div className="feedback-actions">
          <button className="primary" onClick={() => respond(primary)} disabled={sent}>
            {question.labels.keep}
          </button>
          <button className="primary" onClick={() => respond("")} disabled={sent}>
            {question.labels.revert}
          </button>
        </div>
        {question.labels.submit && (
          <div className="feedback-followup">
            <label className="field">
              <span>{question.labels.input}</span>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={2}
              />
            </label>
            <div className="feedback-actions">
              <button
                disabled={sent || !text.trim()}
                onClick={() => respond(text.trim())}
              >
                {question.labels.submit}
              </button>
            </div>
          </div>
        )}
      </>
    );
  } else if (question.widget === "fanout_confirm" || question.widget === "confirm") {
    decision = (
      <div className="feedback-actions">
        <button onClick={() => respond("no")} disabled={sent}>
          {question.labels.cancel}
        </button>
        <button className="primary" onClick={() => respond("y")} disabled={sent}>
          {question.labels.confirm}
        </button>
      </div>
    );
  } else if (isPicker) {
    const pick = question.judge_pick;
    decision = (
      <>
        <p className="feedback-select">
          {subject && <span>{question.labels.select}</span>}
          {hint && <span className="caption">{hint}</span>}
        </p>
        <div className="feedback-actions">
          <button
            className="primary"
            disabled={sent || choice === null}
            onClick={() => respond(String(choice))}
          >
            {question.labels.use}
          </button>
          <button
            className="success"
            disabled={sent}
            onClick={() => respond("")}
            title={pick == null ? hint : `${hint} — ${VOCAB.names.candidate} ${pick}`}
          >
            {question.labels.accept}
          </button>
        </div>
        {question.labels.input && (
          <div className="feedback-followup">
            <label className="field">
              <span>{question.labels.input}</span>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => e.stopPropagation()}
                rows={2}
              />
            </label>
            <div className="feedback-actions">
              <button disabled={sent || !text.trim()} onClick={() => respond(text.trim())}>
                {question.labels.submit}
              </button>
            </div>
          </div>
        )}
      </>
    );
  } else {
    // generic / dataset_description / code_review
    decision = (
      <>
        <label className="field">
          <span>{question.labels.input}</span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              // Enter sends: text is the feedback, empty accepts as-is.
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                respond(text.trim());
              }
            }}
            placeholder={hint ? `${hint} · Shift+Enter for a new line` : undefined}
            rows={3}
            ref={focusRef as React.RefObject<HTMLTextAreaElement>}
          />
        </label>
        <div className="feedback-actions">
          <button
            className="primary"
            disabled={sent || !text.trim()}
            onClick={() => respond(text.trim())}
          >
            {question.labels.submit}
          </button>
          <button className="primary" disabled={sent} onClick={() => respond("")} title={hint}>
            {question.labels.accept}
          </button>
          {question.labels.revert_repair && (
            <button disabled={sent} onClick={() => respond("revert")}>
              {question.labels.revert_repair}
            </button>
          )}
        </div>
      </>
    );
  }

  return (
    <div
      className="feedback-panel"
      tabIndex={-1}
      ref={isPicker || question.widget === "keep_revert" || question.widget === "confirm"
        || question.widget === "fanout_confirm"
        ? (focusRef as React.RefObject<HTMLDivElement>) : undefined}
      onKeyDown={(e) => {
        // Enter anywhere in the panel is the console's Enter: accept as-is
        // (the judge's pick on a picker). A click on the plan to scroll it
        // moves the focus off the feedback box, and Enter must still work.
        // Buttons keep their own Enter (it clicks them); the box handles
        // its own (text is the feedback), and Shift+Enter is a new line.
        const tag = (e.target as HTMLElement).tagName;
        if (e.key === "Enter" && !e.shiftKey && tag !== "BUTTON" && tag !== "TEXTAREA"
            && question.widget !== "keep_revert" && question.widget !== "confirm"
            && question.widget !== "fanout_confirm") {
          e.preventDefault();
          respond("");
        }
      }}
    >
      <div className="feedback-body">{body}</div>
      <div className="feedback-decision">{decision}</div>
    </div>
  );
}
