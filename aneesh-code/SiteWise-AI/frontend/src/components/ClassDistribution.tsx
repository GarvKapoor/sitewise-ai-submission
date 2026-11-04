import type { PredictionResponse } from '../types';

interface Props {
  result: PredictionResponse;
}

/** Land-cover class distribution bars driven by actual model pixel counts. */
export default function ClassDistribution({ result }: Props) {
  const seg = result.segmentation;
  if (!seg) return null;

  const detected = seg.class_percentages.filter((c) => c.percent > 0.001);
  const deleted = seg.class_percentages.length - detected.length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
          Land-cover class distribution
        </h3>
        <span className="text-xs text-slate-500">
          {seg.total_pixels.toLocaleString()} valid pixels · mean confidence{' '}
          {seg.confidence ? `${seg.confidence.mean.toFixed(1)}%` : 'n/a'}
        </span>
      </div>

      {detected.length === 0 && (
        <p className="rounded-lg border border-white/10 bg-navy-850 p-4 text-sm text-slate-400">
          The model assigned no valid pixels.
        </p>
      )}

      {detected.map((c) => (
        <div key={c.class_id}>
          <div className="mb-1 flex items-baseline justify-between text-xs">
            <span className="flex items-center gap-2 font-semibold text-slate-200">
              <span className="h-3 w-3 rounded-[3px]" style={{ backgroundColor: c.color }} />
              {c.name}
              {c.class_id === seg.dominant_class?.class_id && (
                <span className="rounded-full bg-cyan-500/15 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-cyan-300">
                  Dominant
                </span>
              )}
            </span>
            <span className="font-mono text-slate-300">
              {c.percent.toFixed(2)}%<span className="ml-2 text-slate-500">{c.pixel_count.toLocaleString()} px</span>
            </span>
          </div>
          <div className="h-2.5 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full transition-all"
              style={{ width: `${Math.max(c.percent, 0.5)}%`, backgroundColor: c.color }}
            />
          </div>
        </div>
      ))}

      {deleted > 0 && (
        <p className="pt-1 text-xs text-slate-500">
          {deleted} of 10 classes received 0% of pixels and are hidden. Formula: class % = (pixels
          assigned to class ÷ total valid pixels) × 100.
        </p>
      )}
    </div>
  );
}
