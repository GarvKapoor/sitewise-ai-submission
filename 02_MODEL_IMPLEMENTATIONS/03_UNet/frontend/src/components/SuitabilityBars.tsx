import type { PredictionResponse } from '../types';

/** Section D — ranked, project-defined suitability screening scores. */
export default function SuitabilityBars({ result }: { result: PredictionResponse }) {
  const sorted = [...result.suitability].sort((a, b) => b.score - a.score);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {result.top_recommendation && (
          <div className="rounded-xl border border-emerald-500/40 bg-gradient-to-r from-emerald-500/15 to-cyan-500/15 p-4">
            <div className="text-[10px] font-bold uppercase tracking-widest text-emerald-300">
              Top preliminary recommendation
            </div>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-xl font-extrabold text-white">
                {result.top_recommendation.category}
              </span>
              <span className="font-mono text-sm text-emerald-300">
                {result.top_recommendation.score.toFixed(1)}
              </span>
            </div>
          </div>
        )}
        <p className="min-w-[200px] flex-1 text-xs text-slate-400">
          Screening engine uses 7 class proportions + 8 features.
        </p>
      </div>

      <div className="space-y-4">
        {sorted.map((s) => (
          <div
            key={s.category}
            className={`rounded-xl border p-4 ${
              s.is_top
                ? 'border-emerald-500/40 bg-emerald-500/5'
                : 'border-white/10 bg-navy-850/50'
            }`}
          >
            <div className="mb-2 flex items-baseline justify-between gap-3">
              <span className="flex items-center gap-2.5">
                <span
                  className={`flex h-6 w-6 items-center justify-center rounded-md text-[11px] font-bold ${
                    s.rank === 1
                      ? 'bg-gradient-to-br from-emerald-500 to-cyan-500 text-navy-950'
                      : 'bg-white/10 text-slate-300'
                  }`}
                >
                  {s.rank}
                </span>
                <span className="font-semibold text-slate-100">{s.category}</span>
                {s.is_top && (
                  <span className="chip text-[10px] border-emerald-500/40 bg-emerald-500/10 uppercase tracking-wider text-emerald-300">
                    Top pick
                  </span>
                )}
              </span>
              <span className="font-mono text-sm font-bold text-cyan-300">
                {s.score.toFixed(1)}
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full bg-white/5">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400"
                style={{ width: `${Math.min(s.score, 100)}%` }}
              />
            </div>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">{s.explanation}</p>
            {s.contributing_factors.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {s.contributing_factors.map((f, i) => (
                  <span
                    key={i}
                    className="rounded-full bg-white/5 px-2 py-0.5 text-[10px] font-medium text-slate-400"
                  >
                    {f}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <p className="mt-4 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 text-[11px] font-medium text-amber-200/90">
        ⚠️ Preliminary screening scores — not construction approvals or legally verified land-use decisions.
      </p>
    </div>
  );
}
