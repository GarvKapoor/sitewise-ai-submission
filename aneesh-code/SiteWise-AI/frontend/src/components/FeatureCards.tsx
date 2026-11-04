import type { VisualFeature } from '../types';

/** Section C — eight RGB features as compact metric cards. */
export default function FeatureCards({ features }: { features: VisualFeature[] }) {
  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
      {features.map((f) => (
        <div key={f.key} className="rounded-lg border border-white/10 bg-navy-850/60 px-3.5 py-2.5">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm font-bold text-white">{f.display}</span>
            <span className="text-[10px] text-slate-500">{f.unit}</span>
          </div>
          <div className="mt-1 truncate text-[10px] font-medium text-slate-400" title={f.name}>
            {f.name}
          </div>
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-cyan-400"
              style={{ width: `${Math.min(Math.max(f.fraction, 0), 1) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
