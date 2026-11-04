import type { ReliabilityHealth } from '../types';

interface Props {
  health: ReliabilityHealth | null;
}

/** Section E — three simplified model cards. */
export default function ModelInfo({ health }: Props) {
  const unet = health?.models.unet_connected ?? false;
  const rel = health?.models.reliability_connected ?? false;

  const MODELS = [
    { name: 'U-Net', role: 'Land-cover segmentation', connected: unet },
    { name: 'Reliability CNN', role: 'Image quality assessment', connected: rel },
    { name: 'Suitability Engine', role: 'Site-use scoring', connected: true },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-3">
      {MODELS.map((m) => (
        <div key={m.name} className="rounded-xl border border-white/10 bg-navy-850/60 p-4 text-center">
          <h3 className="text-lg font-bold text-white">{m.name}</h3>
          <p className="mt-1 text-sm text-slate-400">{m.role}</p>
          <span
            className={`chip mx-auto mt-3 text-[10px] ${
              m.connected
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                : 'border-amber-500/40 bg-amber-500/10 text-amber-300'
            }`}
          >
            {m.connected ? 'Connected' : 'Not connected'}
          </span>
        </div>
      ))}
    </div>
  );
}
