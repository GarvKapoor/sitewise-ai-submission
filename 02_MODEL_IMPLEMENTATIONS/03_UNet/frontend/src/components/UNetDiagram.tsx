/**
 * Simple, accurate illustration of the U-Net architecture:
 * contracting path (encoder), bottleneck, expanding path (decoder) and
 * skip connections feeding feature concatenation to the reconstruction.
 */

interface Stage {
  label: string;
  depth: string;
  channels: string;
}

const ENCODER: Stage[] = [
  { label: 'Conv 3×3 ×2 + ReLU', depth: '256×256', channels: '32→64' },
  { label: 'Max pool ↓2', depth: '128×128', channels: '→64' },
  { label: 'Conv 3×3 ×2 + ReLU', depth: '128×128', channels: '64→128' },
  { label: 'Max pool ↓2', depth: '64×64', channels: '→128' },
  { label: 'Conv 3×3 ×2 + ReLU', depth: '64×64', channels: '128→256' },
  { label: 'Max pool ↓2', depth: '32×32', channels: '→256' },
];

const DECODER: Stage[] = [
  { label: 'Conv 3×3 ×2 + ReLU', depth: '32×32', channels: '256→512' },
  { label: 'ConvUp ↑2', depth: '64×64', channels: '512→256' },
  { label: 'Concat + Conv 3×3 ×2', depth: '64×64', channels: '512→256' },
  { label: 'ConvUp ↑2', depth: '128×128', channels: '256→128' },
  { label: 'Concat + Conv 3×3 ×2', depth: '128×128', channels: '256→128' },
  { label: 'ConvUp ↑2', depth: '256×256', channels: '128→64' },
  { label: 'Concat + Conv 3×3 ×2', depth: '256×256', channels: '128→64' },
  { label: '1×1 Conv → Softmax', depth: '256×256', channels: '→7 classes' },
];

function Chip({ s, tone, wide }: { s: Stage; tone: string; wide?: boolean }) {
  return (
    <div
      className={`rounded-xl border px-3 py-2 ${tone}`}
      style={{ minWidth: wide ? 190 : 165 }}
    >
      <div className="text-[11px] font-semibold text-slate-100">{s.label}</div>
      <div className="mt-0.5 font-mono text-[10px] text-slate-400">
        {s.depth} · {s.channels}
      </div>
    </div>
  );
}

export default function UNetDiagram() {
  return (
    <div className="glass-card overflow-x-auto p-5">
      <div className="flex min-w-[760px] items-stretch gap-3">
        {/* Encoder */}
        <div className="flex w-[190px] flex-none flex-col justify-between">
          <div className="mb-3 rounded-lg bg-emerald-500/15 px-3 py-1.5 text-center text-xs font-bold uppercase tracking-wider text-emerald-300 ring-1 ring-emerald-400/30">
            Encoder (contracting)
          </div>
          <div className="flex flex-1 flex-col gap-2">
            {ENCODER.map((s, i) => (
              <Chip key={i} s={s} tone="border-emerald-400/25 bg-emerald-500/10" />
            ))}
          </div>
        </div>

        {/* Skip connections column */}
        <div className="relative w-[92px] flex-none">
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2">
            <svg width="92" height="100%" viewBox="0 0 92 400" preserveAspectRatio="none" className="h-full">
              {[9, 43, 77].map((y, i) => (
                <path
                  key={i}
                  d={`M 4 ${y} H 88`}
                  stroke="#22d3ee"
                  strokeWidth="2"
                  strokeDasharray="5 4"
                  opacity="0.9"
                />
              ))}
            </svg>
          </div>
          <div className="absolute left-1/2 top-4 h-[calc(100%-2rem)]">
            {[[9, 'Skip · 64'], [43, 'Skip · 128'], [77, 'Skip · 256']].map(([y, label]) => (
              <div
                key={String(label)}
                className="absolute -translate-y-1/2 whitespace-nowrap rounded-full border border-cyan-400/40 bg-navy-850 px-2 py-0.5 text-[9px] font-semibold text-cyan-300"
                style={{ top: y }}
              >
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Decoder */}
        <div className="flex w-[190px] flex-none flex-col justify-between">
          <div className="mb-3 rounded-lg bg-cyan-500/15 px-3 py-1.5 text-center text-xs font-bold uppercase tracking-wider text-cyan-300 ring-1 ring-cyan-400/30">
            Decoder (expanding)
          </div>
          <div className="flex flex-1 flex-col gap-2">
            {DECODER.map((s, i) => (
              <Chip
                key={i}
                s={s}
                tone={i === DECODER.length - 1 ? 'border-fuchsia-400/40 bg-fuchsia-500/10' : 'border-cyan-400/25 bg-cyan-500/10'}
              />
            ))}
          </div>
        </div>

        {/* Bottleneck / output */}
        <div className="flex w-[170px] flex-none flex-col justify-between">
          <div className="mb-3 rounded-lg bg-fuchsia-500/15 px-3 py-1.5 text-center text-xs font-bold uppercase tracking-wider text-fuchsia-300 ring-1 ring-fuchsia-400/30">
            Output
          </div>
          <div className="flex flex-1 flex-col justify-center gap-3">
            <div className="rounded-xl border border-fuchsia-400/30 bg-fuchsia-500/10 px-3 py-3">
              <div className="text-[11px] font-bold text-fuchsia-200">Bottleneck</div>
              <div className="mt-1 text-[10px] leading-relaxed text-slate-400">
                Deepest feature representation capturing high-level spatial context before reconstruction.
              </div>
            </div>
            <div className="rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 p-3 text-navy-950">
              <div className="text-[11px] font-extrabold">Segmentation mask</div>
              <div className="mt-1 text-[10px] font-semibold leading-relaxed opacity-80">
                Pixel-wise class map · one land-cover class assigned to every pixel
              </div>
            </div>
          </div>
        </div>
      </div>
      <p className="mt-4 text-xs text-slate-500">
        Channel/depth values shown for a 256×256 input with a base of 32 filters and the 7 DeepGlobe
        land-cover output classes. Actual dimensions come from the trained model configuration.
      </p>
    </div>
  );
}
