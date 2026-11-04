export interface ProgressStep {
  key: string;
  label: string;
}

/** 4-step simplified pipeline (Demonstration-friendly). */
export const PIPELINE_STEPS: ProgressStep[] = [
  { key: 'upload', label: 'Upload Image' },
  { key: 'unet', label: 'U-Net Segmentation' },
  { key: 'features', label: 'Land-Cover Analysis' },
  { key: 'suit', label: 'Site Suitability' },
];

export default function AnalysisProgress({ currentStep }: { currentStep: number }) {
  return (
    <div className="glass-card p-6">
      <div className="flex items-center gap-4">
        <span className="relative flex h-10 w-10 items-center justify-center">
          <span className="absolute h-10 w-10 animate-spin-slow rounded-full border-2 border-emerald-500/30 border-t-cyan-400" />
        </span>
        <div>
          <h3 className="text-lg font-bold text-white">Analyzing…</h3>
          <p className="text-sm text-slate-400">Running the 4-step pipeline</p>
        </div>
      </div>

      <div className="mt-5 grid gap-2 sm:grid-cols-4">
        {PIPELINE_STEPS.map((s, i) => {
          const state = i < currentStep ? 'done' : i === currentStep ? 'active' : 'todo';
          return (
            <div
              key={s.key}
              className={`rounded-xl border p-3 text-center text-xs font-semibold ${
                state === 'done'
                  ? 'border-emerald-400/40 bg-emerald-500/10 text-emerald-300'
                  : state === 'active'
                    ? 'border-cyan-400/50 bg-cyan-500/10 text-cyan-200'
                    : 'border-white/10 bg-white/[0.03] text-slate-500'
              }`}
            >
              <span className="block text-[10px] opacity-70">{i + 1}</span>
              {s.label}
            </div>
          );
        })}
      </div>
    </div>
  );
}
