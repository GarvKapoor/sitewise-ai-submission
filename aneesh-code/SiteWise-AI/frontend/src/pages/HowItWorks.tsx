import type { Route } from '../App';
import UNetDiagram from '../components/UNetDiagram';
import { RELIABILITY_THRESHOLD } from '../constants';

const STEPS = [
  {
    n: '01',
    title: 'Upload Image',
    body: 'Select or drag & drop a JPG/PNG satellite image. The frontend previews it and shows its dimensions.',
  },
  {
    n: '02',
    title: 'U-Net Segmentation',
    body: 'The backend preprocesses (RGB → resize → normalise → tensor) and runs the trained U-Net: encoder → bottleneck → decoder with skip connections → pixel-wise softmax mask, one DeepGlobe land-cover class per pixel.',
  },
  {
    n: '03',
    title: 'Land-Cover Analysis',
    body: 'Class percentages come straight from predicted pixel counts. Eight RGB features are extracted, refined by segmentation where compatible. A separate Reliability CNN checks image quality against the project threshold.',
  },
  {
    n: '04',
    title: 'Site Suitability',
    body: 'A deterministic engine combines class proportions + features and ranks eight preliminary site-use categories with short explanations.',
  },
];

export default function HowItWorks({ onNavigate }: { onNavigate: (to: Route) => void }) {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <header className="text-center">
        <span className="chip border-cyan-500/30 bg-cyan-500/10 text-cyan-300">How It Works</span>
        <h1 className="mt-4 text-4xl font-extrabold text-white">
          From satellite image to <span className="accent-gradient-text">ranked shortlist</span>
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-slate-400">
          Four steps. The central output is a pixel-wise land-cover map — not a single class label.
        </p>
      </header>

      {/* U-Net architecture */}
      <section className="mt-12">
        <h2 className="section-heading">U-Net architecture</h2>
        <div className="mt-6">
          <UNetDiagram />
        </div>
      </section>

      {/* Four steps */}
      <section className="mt-12 space-y-4">
        <h2 className="section-heading">The 4-step workflow</h2>
        {STEPS.map((s) => (
          <div key={s.n} className="glass-card flex gap-4 p-5 sm:p-6">
            <span className="font-mono text-2xl font-extrabold text-cyan-400/70">{s.n}</span>
            <div>
              <h3 className="text-lg font-bold text-white">{s.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-400">{s.body}</p>
            </div>
          </div>
        ))}
      </section>

      {/* Reliability note */}
      <section className="glass-card mt-8 p-6">
        <h3 className="text-lg font-bold text-white">Reliability ≠ segmentation confidence</h3>
        <p className="mt-2 text-sm leading-relaxed text-slate-400">
          The Reliability CNN scores image quality against the {RELIABILITY_THRESHOLD}% threshold. A
          reliable image does not automatically mean the segmentation is correct — the two scores are
          reported independently.
        </p>
      </section>

      <div className="mt-10 text-center">
        <button className="btn-primary" onClick={() => onNavigate('analyze')}>
          Analyze Your Site
        </button>
      </div>
    </div>
  );
}
