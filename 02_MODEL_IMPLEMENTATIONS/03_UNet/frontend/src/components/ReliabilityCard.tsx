import type { PredictionResponse } from '../types';

/** Section A widget — image reliability (separate CNN from the U-Net). */
export default function ReliabilityCard({ result }: { result: PredictionResponse }) {
  const rel = result.reliability;
  if (!rel) {
    return (
      <div className="rounded-xl border border-white/10 bg-navy-850 p-5 text-sm text-slate-400">
        Reliability assessment unavailable.
      </div>
    );
  }

  const statusTone =
    rel.status === 'clean'
      ? 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10'
      : rel.status === 'degraded'
        ? 'text-red-300 border-red-500/40 bg-red-500/10'
        : 'text-slate-300 border-white/20 bg-white/5';

  return (
    <div className="rounded-xl border border-white/10 bg-navy-850/60 p-5">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
          Image reliability
        </h3>
        <span className={`chip font-bold uppercase tracking-wider ${statusTone}`}>
          {rel.status === 'clean' ? 'Clean' : rel.status === 'degraded' ? 'Degraded' : 'Unavailable'}{' '}
          · {rel.passed ? 'Pass' : rel.status === 'unavailable' ? '—' : 'Fail'}
        </span>
      </div>

      <div className="mt-4 flex items-end gap-3">
        <span className="text-4xl font-extrabold text-white">{rel.score.toFixed(1)}%</span>
        <span className="pb-1 text-xs text-slate-400">
          of {rel.threshold}% threshold
        </span>
      </div>
      <div className="relative mt-2 h-3 overflow-hidden rounded-full bg-white/5">
        <div
          className={`h-full rounded-full ${rel.passed ? 'bg-gradient-to-r from-emerald-500 to-cyan-400' : 'bg-gradient-to-r from-red-500 to-orange-400'}`}
          style={{ width: `${Math.min(rel.score, 100)}%` }}
        />
        <div
          className="absolute bottom-0 top-0 w-px bg-white/70"
          style={{ left: `${rel.threshold}%` }}
          title={`Reliability threshold: ${rel.threshold}%`}
        />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-lg bg-navy-800/70 p-3">
          <dt className="text-slate-500">Quality status</dt>
          <dd className="mt-0.5 font-semibold text-slate-200">
            {rel.passed ? 'Accepted for screening' : 'Below threshold — interpret with caution'}
          </dd>
        </div>
        <div className="rounded-lg bg-navy-800/70 p-3">
          <dt className="text-slate-500">Assessed by</dt>
          <dd className="mt-0.5 font-semibold text-slate-200">
            {rel.model_used ? 'Reliability CNN (separate model)' : 'Heuristic fallback (labelled)'}
          </dd>
        </div>
      </dl>

      <p className="mt-3 text-[11px] text-amber-200/90">
        Reliability measures image quality only — not segmentation correctness.
      </p>
    </div>
  );
}
