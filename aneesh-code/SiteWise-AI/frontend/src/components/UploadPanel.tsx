import { useCallback, useRef, useState } from 'react';

export interface UploadState {
  file: File;
  previewUrl: string;
  width: number;
  height: number;
}

interface Props {
  upload: UploadState | null;
  busy: boolean;
  onSelect: (upload: UploadState) => void;
  onRemove: () => void;
}

const ACCEPTED = ['image/jpeg', 'image/jpg', 'image/png'];

export default function UploadPanel({ upload, busy, onSelect, onRemove }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadFile = useCallback(
    (file: File) => {
      setError(null);
      if (!ACCEPTED.includes(file.type)) {
        setError('Unsupported file type — upload a JPG, JPEG or PNG satellite image.');
        return;
      }
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        onSelect({ file, previewUrl: url, width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = () => {
        setError('Could not read this image file.');
        URL.revokeObjectURL(url);
      };
      img.src = url;
    },
    [onSelect],
  );

  return (
    <div>
      {!upload ? (
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files?.[0];
            if (file) loadFile(file);
          }}
          className={`rounded-2xl border-2 border-dashed p-8 text-center transition sm:p-12 ${
            dragOver
              ? 'border-cyan-400 bg-cyan-500/10'
              : 'border-white/15 bg-white/[0.02] hover:border-emerald-400/50'
          }`}
        >
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 ring-1 ring-cyan-400/30">
            <svg viewBox="0 0 24 24" className="h-8 w-8 text-cyan-300" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 16.2 9.5 11l4 4.5 3-3.5L20 16" />
              <circle cx="16.5" cy="6.5" r="2" />
              <rect x="3" y="3" width="18" height="18" rx="4" />
            </svg>
          </div>
          <h3 className="mt-4 text-lg font-bold text-white">Drag & drop your satellite image</h3>
          <p className="mt-1 text-sm text-slate-400">JPG, JPEG or PNG · aerial or satellite imagery</p>
          <button className="btn-primary mt-5" onClick={() => inputRef.current?.click()}>
            Browse files
          </button>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadFile(f);
              e.target.value = '';
            }}
          />
        </div>
      ) : (
        <div className="rounded-2xl border border-white/10 bg-navy-850/60 p-4">
          <div className="flex flex-col gap-4 sm:flex-row">
            <div className="overflow-hidden rounded-xl border border-white/10 bg-navy-950 sm:w-64">
              <img src={upload.previewUrl} alt="Upload preview" className="max-h-52 w-full object-contain" />
            </div>
            <div className="flex flex-1 flex-col justify-center gap-2 text-sm">
              <div className="flex items-baseline justify-between gap-3">
                <span className="truncate font-semibold text-white" title={upload.file.name}>
                  {upload.file.name}
                </span>
                <span className="font-mono text-xs text-slate-400">
                  {upload.width} × {upload.height} px
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span className="chip border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
                  Ready
                </span>
                <span>{(upload.file.size / 1024).toFixed(0)} KB</span>
                <span>·</span>
                <span>{uploadFileStatus(busy)}</span>
              </div>
              <div className="mt-2 flex gap-2">
                <button className="btn-secondary !px-4 !py-2 text-sm" onClick={onRemove} disabled={busy}>
                  Remove
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <p className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
          {error}
        </p>
      )}

    </div>
  );
}

function uploadFileStatus(busy: boolean) {
  return busy ? 'Processing…' : 'Awaiting analysis';
}
