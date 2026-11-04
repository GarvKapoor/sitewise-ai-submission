import type { Route } from '../App';
import { LAND_COVER_CLASSES } from '../constants';

const CATEGORIES = [
  'Residential Development',
  'Commercial Development',
  'Industrial Development',
  'Educational Development',
  'Healthcare Development',
  'Agricultural Use',
  'Transportation Infrastructure',
  'Green / Conservation',
];

export default function About({ onNavigate }: { onNavigate: (to: Route) => void }) {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6">
      <header className="text-center">
        <span className="chip border-cyan-500/30 bg-cyan-500/10 text-cyan-300">About</span>
        <h1 className="mt-4 text-4xl font-extrabold text-white">
          SiteWise <span className="accent-gradient-text">AI</span>
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-slate-400">
          A deep-learning project that segments satellite land cover with U-Net and ranks
          preliminary site-use suitability.
        </p>
      </header>

      <section className="mt-8 grid gap-4 sm:grid-cols-2">
        <div className="glass-card p-6">
          <h2 className="text-lg font-bold text-white">How it works</h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-400">
            A U-Net assigns one of seven DeepGlobe land-cover classes to every pixel. Class
            percentages + eight RGB features feed a suitability engine that ranks eight categories.
            A separate Reliability CNN screens image quality in parallel.
          </p>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            {LAND_COVER_CLASSES.map((c) => (
              <span key={c.id} className="flex items-center gap-1.5 text-slate-300">
                <span className="h-3 w-3 rounded-[3px]" style={{ backgroundColor: c.color }} />
                {c.name}
              </span>
            ))}
          </div>
        </div>
        <div className="glass-card p-6">
          <h2 className="text-lg font-bold text-white">Project facts</h2>
          <dl className="mt-3 space-y-2 text-sm">
            {[
              ['Dataset', 'DeepGlobe Land Cover (pixel masks)'],
              ['Architecture', 'U-Net encoder–decoder'],
              ['Backend', 'Python Flask'],
              ['Framework', 'TensorFlow / Keras'],
              ['Extra model', 'Reliability CNN'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3 border-b border-white/5 pb-1.5">
                <dt className="text-slate-500">{k}</dt>
                <dd className="font-medium text-slate-200">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section className="glass-card mt-4 p-6">
        <h2 className="text-lg font-bold text-white">Eight screening categories</h2>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {CATEGORIES.map((c, i) => (
            <span key={c} className="rounded-full border border-emerald-400/20 bg-emerald-500/10 px-3 py-1 font-medium text-emerald-200">
              {i + 1}. {c}
            </span>
          ))}
        </div>
      </section>

      <section className="glass-card mt-4 border-amber-500/20 bg-amber-500/[0.04] p-5">
        <p className="text-sm leading-relaxed text-amber-200/90">
          Recommendations are preliminary, project-defined screening results — not construction
          approvals or legally verified land-use decisions.
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
