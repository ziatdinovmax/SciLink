import { useEffect, useRef } from "react";

/** Two-item picker menu — "files" or "folder" — behind a single upload
 * control. A native file input opens EITHER the file dialog or (with
 * `webkitdirectory`) the directory dialog, never one that offers both, so
 * one visible button routes through this menu to two hidden inputs.
 * Closes on outside click or Escape. */
export function UploadMenu({
  onFiles,
  onFolder,
  onClose,
  placement = "below",
}: {
  onFiles: () => void;
  onFolder: () => void;
  onClose: () => void;
  placement?: "below" | "above";
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    // Deferred so the click that opened the menu is not the one closing it.
    const t = setTimeout(() => {
      document.addEventListener("mousedown", onDown);
      document.addEventListener("keydown", onKey);
    }, 0);
    return () => {
      clearTimeout(t);
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  return (
    <div
      ref={ref}
      className={`upload-menu ${placement}`}
      role="menu"
      onClick={(e) => e.stopPropagation()}
    >
      <button type="button" role="menuitem" onClick={() => { onClose(); onFiles(); }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round"
          strokeLinejoin="round" aria-hidden="true">
          <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
          <path d="M14 3v6h6" />
        </svg>
        Upload files
      </button>
      <button type="button" role="menuitem" onClick={() => { onClose(); onFolder(); }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round"
          strokeLinejoin="round" aria-hidden="true">
          <path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
        </svg>
        Upload folder
        <span className="upload-menu-hint">subfolders included</span>
      </button>
    </div>
  );
}
