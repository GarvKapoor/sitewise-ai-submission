import { useState } from 'react';
import type { PredictionResponse } from '../types';
import BeforeAfterSlider from './BeforeAfterSlider';

type ViewMode = 'original' | 'mask' | 'overlay';

const TABS: { id: ViewMode; label: string }[] = [
  { id: 'original', label: 'Original Image' },
  { id: 'mask', label: 'Segmentation Mask' },
  { id: 'overlay', label: 'Overlay' },
];

interface Props {
  result: PredictionResponse;
}

/** The centrepiece of the results dashboard: original / mask / overlay switcher
 *  plus an interactive before-after comparison slider. */
export default function SegmentationViewer({ result }: Props) {
  const [mode, setMode] = useState<ViewMode>('overlay');
  const seg = result.segmentation;

  if (!seg) {
    return (
      <div className="rounded-xl border border-white/10 bg-navy-850 p-6 text-center text-sm text-slate-400">
        No segmentation output available.
      </div>
    );
  }

  // Demonstration labelling is driven by the segmentation's own provenance,
  // not by a global flag — a missing Reliability CNN must never relabel
  // genuine U-Net output as demo data.
  const demo = seg.source !== 'unet';
  const src =
    mode === 'original'
      ? result.image.original_base64
      : mode === 'mask'
        ? seg.mask_base64
        : seg.overlay_base64;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setMode(t.id)}
            className={`rounded-lg px-4 py-2 text-sm font-semibold transition ${
              mode === t.id
                ? 'bg-gradient-to-r from-emerald-500/20 to-cyan-500/20 text-white ring-1 ring-cyan-400/40'
                : 'bg-white/5 text-slate-300 hover:bg-white/10 hover:text-white'
            }`}
          >
            {t.label}
          </button>
        ))}
        {seg.source === 'demonstration_heuristic' && (
          <span className="chip text-[10px] border-amber-500/40 bg-amber-500/10 font-bold uppercase tracking-wider text-amber-300">
            Demonstration
          </span>
        )}
      </div>

      <div className="relative">
        <img src={src} alt={`${TABS.find((t) => t.id === mode)!.label} view`} className="mx-auto max-h-[560px] w-full rounded-xl border border-white/10 bg-navy-950 object-contain" />
      </div>

      <div className="mt-4">
        <BeforeAfterSlider original_base64={result.image.original_base64} overlay_base64={seg.overlay_base64} demo={demo} />
      </div>
    </div>
  );
}
