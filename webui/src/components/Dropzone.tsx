import { useRef, useState } from "react";
import type { FolderEntry } from "../api";
import { groupDirectoryInput, splitDrop } from "../folderfiles";
import { UploadMenu } from "./UploadMenu";

/** Click-or-drop file picker that uploads immediately via the callback and
 * shows the accumulated file names as chips.
 *
 * With `onFolder` set, the zone also takes folders: a dropped directory is
 * walked recursively (subfolders included), and a click opens a files /
 * folder menu instead of the file dialog directly (the browser has no
 * single dialog for both). Each top-level folder goes to `onFolder` as
 * one batch; loose files keep going to `onFiles`. */
export function Dropzone({
  label,
  accept,
  multiple = true,
  onFiles,
  onFolder,
}: {
  label: string;
  accept?: string;
  multiple?: boolean;
  onFiles: (files: File[]) => Promise<string[]>; // returns saved names
  onFolder?: (name: string, entries: FolderEntry[]) => Promise<string[]>; // returns chip labels
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const dirRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [names, setNames] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(false);

  const handle = async (list: FileList | null) => {
    if (!list || list.length === 0) return;
    setError(null);
    try {
      const saved = await onFiles(Array.from(list));
      setNames((prev) => [...prev, ...saved]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleFolders = async (folders: Map<string, FolderEntry[]>) => {
    if (!onFolder || folders.size === 0) return;
    setError(null);
    setBusy(true);
    try {
      for (const [name, entries] of folders) {
        const saved = await onFolder(name, entries);
        setNames((prev) => [...prev, ...saved]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onDrop = async (dt: DataTransfer) => {
    if (!onFolder) {
      void handle(dt.files);
      return;
    }
    const split = await splitDrop(dt);
    if (split.files.length) {
      const list = new DataTransfer();
      for (const f of split.files) list.items.add(f);
      await handle(list.files);
    }
    await handleFolders(split.folders);
  };

  return (
    <div className="dropzone-wrap">
      <div
        className={`dropzone${drag ? " drag" : ""}`}
        onClick={() => (onFolder ? setMenu((m) => !m) : inputRef.current?.click())}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          void onDrop(e.dataTransfer);
        }}
      >
        {busy ? "Uploading folder…" : label}
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={(e) => {
            void handle(e.target.files);
            e.target.value = "";
          }}
        />
        {onFolder && (
          <input
            ref={dirRef}
            type="file"
            // Non-standard attribute (all major browsers honor it); not in
            // React's typed props, hence the spread.
            {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
            multiple
            onChange={(e) => {
              void handleFolders(groupDirectoryInput(e.target.files));
              e.target.value = "";
            }}
          />
        )}
        {names.length > 0 && (
          <div className="file-chips">
            {names.map((n, i) => (
              <span className="file-chip" key={`${n}-${i}`}>
                {n}
              </span>
            ))}
          </div>
        )}
      </div>
      {menu && onFolder && (
        <UploadMenu
          onFiles={() => inputRef.current?.click()}
          onFolder={() => dirRef.current?.click()}
          onClose={() => setMenu(false)}
        />
      )}
      {error && <p className="caption" style={{ color: "var(--danger)" }}>{error}</p>}
    </div>
  );
}
