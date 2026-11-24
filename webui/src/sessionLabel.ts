/** Short names for what the Workspace panel lists: a session by when it
 * started, a model without its provider prefix; and the rule that keeps two
 * rows of one list from sharing a name. */

const PREFIX = /^[a-z_]+_session_/;
// <prefix>_<YYYYmmdd>_<HHMMSS>, and the _2, _3 ... a name already taken gets
const STAMP = /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(?:_(\d+))?$/;

/** A session id as a person reads it: its start time ("Nov 22, 20:15") when
 * the id carries one, else the id without its mode prefix. The digits are the
 * server's clock and are shown as written: no time zone or daylight-saving
 * rule of the browser moves them. */
export function sessionLabel(sid: string, seconds = false): string {
  if (sid === "unattributed") return "not in a session";
  const m = sid.match(STAMP);
  const plain = sid.replace(PREFIX, "");
  if (!m) return plain;
  const [y, mo, d, h, mi, s] = m.slice(1, 7).map(Number);
  const t = new Date(Date.UTC(y, mo - 1, d, h, mi, s));
  // digits that are not a date (month 13) roll over into another one
  if (Number.isNaN(t.getTime()) || t.getUTCMonth() !== mo - 1 || t.getUTCDate() !== d
      || t.getUTCHours() !== h || t.getUTCMinutes() !== mi) return plain;
  const text = t.toLocaleString(undefined, {
    timeZone: "UTC", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    second: seconds ? "2-digit" : undefined, hourCycle: "h23",
  });
  return m[7] ? `${text} (${m[7]})` : text;
}

/** A model id without its provider prefix. */
export function modelLabel(model: string): string {
  return model.replace(/^[a-z_]+\//, "");
}

/** One label per id: the first of `forms` under which no other id of the
 * list reads the same, else the id itself. */
export function distinctLabels(ids: string[], forms: Array<(id: string) => string>): Map<string, string> {
  const out = new Map<string, string>();
  for (const form of [...forms, (id: string) => id]) {
    const count = new Map<string, number>();
    for (const id of ids) count.set(form(id), (count.get(form(id)) ?? 0) + 1);
    for (const id of ids) if (!out.has(id) && count.get(form(id)) === 1) out.set(id, form(id));
  }
  for (const id of ids) if (!out.has(id)) out.set(id, id);
  return out;
}

export const sessionLabels = (ids: string[]) =>
  distinctLabels(ids, [(id) => sessionLabel(id), (id) => sessionLabel(id, true)]);

export const modelLabels = (ids: string[]) => distinctLabels(ids, [modelLabel]);
