import { useRef, useState } from "react";

/** Click-or-drop file picker that uploads immediately via the callback and
 * shows the accumulated file names as chips. */
export function Dropzone({
  label,
  accept,
  multiple = true,
  onFiles,
}: {
  label: string;
  accept?: string;
  multiple?: boolean;
  onFiles: (files: File[]) => Promise<string[]>; // returns saved names
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [names, setNames] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div>
      <div
        className={`dropzone${drag ? " drag" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          void handle(e.dataTransfer.files);
        }}
      >
        {label}
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
      {error && <p className="caption" style={{ color: "var(--danger)" }}>{error}</p>}
    </div>
  );
}
