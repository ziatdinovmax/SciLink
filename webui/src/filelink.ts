/** Make file paths mentioned in chat text clickable: decide whether an
 * inline-code token looks like a file reference, and resolve it against the
 * session's real tree (agents write paths absolute, "…"-elided, or as bare
 * filenames, so the text alone can't be trusted as a path). */

import { api, type TreeEntry } from "./api";

// Known session-file extensions — a whitelist, so code-ish inline tokens
// like `np.load` or `plt.savefig` don't read as file references.
const FILE_EXTS = new Set([
  "png", "jpg", "jpeg", "tif", "tiff", "npy", "csv", "tsv", "xlsx",
  "json", "jsonl", "md", "html", "htm", "pdf", "txt", "log", "py",
  "yaml", "yml", "h5", "hdf5", "nxs", "cif", "xyz", "sh", "zip",
]);

/** Path-like: no newlines, reasonable length, ends in a known extension. */
export function isFileToken(token: string): boolean {
  const t = token.trim();
  if (t.length <= 3 || t.length >= 300 || t.includes("\n")) return false;
  const m = /\.([A-Za-z0-9]{1,6})$/.exec(t);
  return m !== null && FILE_EXTS.has(m[1].toLowerCase());
}

function flatten(entries: TreeEntry[]): TreeEntry[] {
  const out: TreeEntry[] = [];
  for (const e of entries) {
    if (e.is_dir) out.push(...flatten(e.children ?? []));
    else out.push(e);
  }
  return out;
}

/** Session-relative path for a mentioned file, or null when nothing in the
 * tree matches. Longest-suffix match first, then basename; ties go to the
 * most recently modified file (the one the message is most likely about). */
export async function resolveSessionPath(
  sessionId: string,
  token: string,
): Promise<string | null> {
  let t = token.trim().replace(/^\.{2,}\/?/, ""); // strip "…/" elision
  const marker = sessionId + "/";
  const i = t.indexOf(marker);
  if (i >= 0) t = t.slice(i + marker.length); // absolute → relative
  t = t.replace(/^\/+/, "").toLowerCase();
  if (!t) return null;

  let files: TreeEntry[];
  try {
    files = flatten((await api.tree(sessionId)).entries);
  } catch {
    return null;
  }
  let candidates = files.filter((f) =>
    ("/" + f.path.toLowerCase()).endsWith("/" + t),
  );
  if (!candidates.length) {
    const base = t.split("/").pop()!;
    candidates = files.filter(
      (f) => f.path.toLowerCase().split("/").pop() === base,
    );
  }
  if (!candidates.length) return null;
  candidates.sort((a, b) => b.mtime - a.mtime);
  return candidates[0].path;
}
