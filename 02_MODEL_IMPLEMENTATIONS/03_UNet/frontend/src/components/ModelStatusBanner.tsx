import type { ReliabilityHealth } from '../types';

interface Props {
  health: ReliabilityHealth | null;
  healthLoading: boolean;
  healthError: string | null;
  onRefresh: () => void;
}

/**
 * Compact single-line backend status with expandable detail. Still shows the
 * "Model Not Connected" state whenever the trained U-Net / reliability models
 * are missing.
 */
export default function ModelStatusBanner({ health, healthLoading, healthError, onRefresh }: Props) {
  if (healthLoading) {
    return (
      <div className="glass-card flex items-center gap-3 px-4 py-2.5 text-sm text-slate-300">
        <span className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-cyan-400/30 border-t-cyan-400" />
        Checking backend…
      </div>
    );
  }

  if (!health) {
    return (
      <div className="glass-card flex items-center gap-3 border-red-500/30 bg-red-500/5 px-4 py-2.5 text-sm">
        <span className="chip text-[10px] border-red-500/40 bg-red-500/10 font-bold uppercase text-red-300">
          Backend offline
        </span>
        <span className="truncate text-slate-400">{healthError ?? 'Run the Flask backend'}</span>
        <button className="ml-auto shrink-0 text-xs font-semibold text-cyan-300 hover:underline" onClick={onRefresh}>
          Retry
        </button>
      </div>
    );
  }

  const states = health.model_status ?? {
    unet: (health.models.unet_connected ? 'connected' : 'unavailable') as 'connected' | 'unavailable',
    reliability: (health.models.reliability_connected ? 'connected' : 'unavailable') as 'connected' | 'unavailable',
  };

  const chipFor = (name: string, state: string) => {
    const tone =
      state === 'connected'
        ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
        : state === 'failed'
          ? 'border-red-500/40 bg-red-500/10 text-red-300'
          : 'border-amber-500/40 bg-amber-500/10 text-amber-300';
    const label =
      state === 'connected' ? 'connected' : state === 'failed' ? 'error' : 'not connected';
    return (
      <span className={`chip text-[10px] font-bold uppercase tracking-wider ${tone}`}>
        {name}: {label}
      </span>
    );
  };

  const allOk = states.unet === 'connected' && states.reliability === 'connected';

  if (allOk) {
    return (
      <div className="glass-card flex items-center gap-3 border-emerald-500/25 px-4 py-2.5 text-sm text-emerald-200">
        <span className="h-2 w-2 rounded-full bg-emerald-400" />
        U-Net + Reliability CNN connected · real model output
      </div>
    );
  }

  return (
    <div className="glass-card border-amber-500/30 bg-amber-500/5 px-4 py-2.5">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        {chipFor('U-Net', states.unet)}
        {chipFor('Reliability CNN', states.reliability)}
        {states.reliability !== 'connected' && (
          <span className="text-xs text-slate-400">Suitability engine: deterministic scoring (always available)</span>
        )}
        <details className="text-xs text-slate-400">
          <summary className="cursor-pointer font-semibold text-cyan-300">How to connect</summary>
          <span className="mt-1 block">
            Drop <code className="rounded bg-navy-850 px-1 text-cyan-300">.keras/.h5</code> files into{' '}
            <code className="rounded bg-navy-850 px-1 text-cyan-300">backend/models/</code> and restart the Flask service.
          </span>
        </details>
        <button className="ml-auto text-xs font-semibold text-cyan-300 hover:underline" onClick={onRefresh}>
          Refresh
        </button>
      </div>
    </div>
  );
}
