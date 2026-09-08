/** Folder pickers: turn a `webkitdirectory` input's FileList, or a drop's
 * DataTransfer, into folder-upload entries grouped by top-level folder.
 *
 * Both browser routes hand us files with a path relative to the picked
 * folder's PARENT (`<folder>/<sub>/<name>`): the directory input via
 * `webkitRelativePath`, the drop via the FileSystem entry API (which is the
 * only way a dropped directory yields anything — `dataTransfer.files` lists
 * a dropped folder as one empty entry). Hidden entries are skipped and a
 * total cap stops a runaway drop of a whole home directory. */

import type { FolderEntry } from "./api";

export const MAX_FOLDER_FILES = 2000; // mirrors files.MAX_FOLDER_FILES

export interface DropSplit {
  files: File[]; // loose files dropped alongside
  folders: Map<string, FolderEntry[]>; // top-level folder name → entries
}

/** Group a directory input's FileList by its top-level folder. */
export function groupDirectoryInput(list: FileList | null): Map<string, FolderEntry[]> {
  const folders = new Map<string, FolderEntry[]>();
  if (!list) return folders;
  for (const file of Array.from(list)) {
    const rel = (file as File & { webkitRelativePath?: string }).webkitRelativePath || "";
    if (!rel.includes("/")) continue; // not from a directory pick
    if (isHidden(rel)) continue;
    const top = rel.slice(0, rel.indexOf("/"));
    const entries = folders.get(top) ?? [];
    if (entries.length >= MAX_FOLDER_FILES) continue;
    entries.push({ file, relPath: rel });
    folders.set(top, entries);
  }
  return folders;
}

/** Split a drop into loose files and folders (walked recursively). Falls
 * back to `dataTransfer.files` when the entry API is unavailable. */
export async function splitDrop(dt: DataTransfer): Promise<DropSplit> {
  const out: DropSplit = { files: [], folders: new Map() };
  const items = Array.from(dt.items ?? []);
  const entries = items
    .map((it) => (typeof it.webkitGetAsEntry === "function" ? it.webkitGetAsEntry() : null))
    .filter((e): e is FileSystemEntry => e !== null);
  if (entries.length === 0) {
    out.files = Array.from(dt.files ?? []);
    return out;
  }
  // Collect File objects up front: DataTransfer items are only readable
  // during the drop event, before the first await.
  const looseFiles = items.map((it) => (it.kind === "file" ? it.getAsFile() : null));
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    if (isHidden(entry.name)) continue;
    if (entry.isDirectory) {
      const collected: FolderEntry[] = [];
      await walkDirectory(entry as FileSystemDirectoryEntry, entry.name, collected);
      if (collected.length) out.folders.set(entry.name, collected);
    } else {
      const f = looseFiles[i];
      if (f) out.files.push(f);
    }
  }
  return out;
}

async function walkDirectory(
  dir: FileSystemDirectoryEntry,
  prefix: string,
  out: FolderEntry[],
): Promise<void> {
  const reader = dir.createReader();
  // readEntries returns batches (Chrome: 100 at a time) until an empty one.
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((resolve, reject) =>
      reader.readEntries(resolve, reject),
    );
    if (batch.length === 0) break;
    for (const entry of batch) {
      if (out.length >= MAX_FOLDER_FILES) return;
      if (isHidden(entry.name)) continue;
      const rel = `${prefix}/${entry.name}`;
      if (entry.isDirectory) {
        await walkDirectory(entry as FileSystemDirectoryEntry, rel, out);
      } else if (entry.isFile) {
        const file = await new Promise<File>((resolve, reject) =>
          (entry as FileSystemFileEntry).file(resolve, reject),
        );
        out.push({ file, relPath: rel });
      }
    }
  }
}

function isHidden(relPath: string): boolean {
  return relPath.split("/").some((c) => c.startsWith("."));
}

/** Short human summary of a folder upload result for chips / notes. */
export function describeFolder(root: string, nFiles: number, nDirs: number): string {
  const name = root.split("/").filter(Boolean).pop() ?? root;
  const sub = nDirs > 1 ? `, ${nDirs - 1} subfolder${nDirs - 1 === 1 ? "" : "s"}` : "";
  return `${name}/ (${nFiles} file${nFiles === 1 ? "" : "s"}${sub})`;
}

/** Prompt fragment describing an uploaded folder's layout for the agent —
 * the agents' directory listings are one level deep, so a nested folder
 * must be spelled out (each subfolder's absolute path and file count). */
export function folderPromptLines(r: {
  root: string;
  dirs: { path: string; n_files: number }[];
  paths: string[];
}): string {
  const subs = r.dirs.filter((d) => d.path !== r.root);
  if (subs.length === 0) {
    return `Folder \`${r.root}\` (${r.paths.length} file${r.paths.length === 1 ? "" : "s"})`;
  }
  const rootCount = r.dirs.find((d) => d.path === r.root)?.n_files ?? 0;
  const listed = subs
    .slice(0, 40)
    .map((d) => `  - \`${d.path}\` (${d.n_files} file${d.n_files === 1 ? "" : "s"})`)
    .join("\n");
  const more = subs.length > 40 ? `\n  - … and ${subs.length - 40} more subfolders` : "";
  return (
    `Folder \`${r.root}\` (${r.paths.length} files; ${rootCount} at the top level, ` +
    `the rest in ${subs.length} subfolder${subs.length === 1 ? "" : "s"}):\n${listed}${more}`
  );
}
