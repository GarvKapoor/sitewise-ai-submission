import { useRef, useState } from 'react';

interface Props {
  original_base64: string;
  overlay_base64: string;
  demo: boolean;
}

/**
 * Interactive before/after comparison: drag the divider to reveal the
 * original satellite image on the right and the segmentation overlay on the
 * left. Includes a keyboard-accessible range control.
 */
export default function BeforeAfterSlider({ original_base64, overlay_base64, demo }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  const update = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos(Math.min(94, Math.max(6, ((clientX - rect.left) / rect.width) * 100)));
  };

  return (
    <div>
      <div
        ref={containerRef}
        className="relative select-none overflow-hidden rounded-xl border border-white/10 bg-navy-950"
        onPointerDown={(e) => {
          dragging.current = true;
          (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
          update(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && update(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerLeave={() => (dragging.current = false)}
      >
        {/* Right side: original */}
        <img
          src={original_base64}
          alt="Original satellite image"
          draggable={false}
          className="block max-h-[520px] w-full object-contain"
        />
        {/* Left side: segmentation overlay */}
        <img
          src={overlay_base64}
          alt="Segmentation overlay"
          draggable={false}
          className="block absolute inset-0 h-full w-full object-contain"
          style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
        />

        {/* Divider */}
        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-cyan-300 shadow-[0_0_8px_rgba(34,211,238,0.8)]"
          style={{ left: `${pos}%` }}
        >
          <div className="absolute left-1/2 top-1/2 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize items-center justify-center rounded-full bg-cyan-300 text-navy-950 shadow-lg">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 7l-5 5 5 5M16 7l5 5-5 5" />
            </svg>
          </div>
        </div>

        <span className="chip pointer-events-none absolute left-3 top-3 border-cyan-400/40 bg-navy-950/80 text-[10px] font-bold uppercase tracking-wider text-cyan-200">
          Segmentation
        </span>
        <span className="chip pointer-events-none absolute right-3 top-3 border-white/20 bg-navy-950/80 text-[10px] font-bold uppercase tracking-wider text-white/80">
          Original
        </span>

        {/* Keyboard control */}
        <input
          type="range"
          min={6}
          max={94}
          value={Math.round(pos)}
          onChange={(e) => setPos(Number(e.target.value))}
          aria-label="Compare original image and segmentation overlay"
          className="absolute bottom-2 left-1/2 h-1 w-2/3 -translate-x-1/2 cursor-pointer appearance-none rounded-full bg-white/20 opacity-60 hover:opacity-100 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-300"
        />
      </div>
      {demo && (
        <p className="mt-2 text-[11px] text-amber-300/90">
          Demonstration overlay from colour heuristics — not a U-Net prediction.
        </p>
      )}
    </div>
  );
}
