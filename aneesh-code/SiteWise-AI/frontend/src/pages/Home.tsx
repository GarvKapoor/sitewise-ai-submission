import type { Route } from '../App';
import type { UploadState } from '../components/UploadPanel';
import UploadPanel from '../components/UploadPanel';
import { setPendingUpload } from '../staging';
import { LAND_COVER_CLASSES } from '../constants';

interface HomeProps {
  onNavigate: (to: Route) => void;
}

export default function Home({ onNavigate }: HomeProps) {
  const handleSelect = (u: UploadState) => {
    setPendingUpload(u);
    onNavigate('analyze');
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6">
      {/* Hero */}
      <section className="text-center">
        <span className="chip border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
          U-Net semantic segmentation
        </span>
        <h1 className="mx-auto mt-4 max-w-3xl text-4xl font-extrabold leading-tight text-white sm:text-5xl">
          Understand Land. <span className="accent-gradient-text">Make Smarter Decisions.</span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-slate-400">
          Upload a satellite image — SiteWise AI segments land cover, assesses reliability
          and ranks preliminary site-use suitability.
        </p>
      </section>

      {/* Prominent upload */}
      <section className="mt-10">
        <UploadPanel upload={null} busy={false} onSelect={handleSelect} onRemove={() => undefined} />
        <p className="mt-4 text-center text-sm text-slate-400">
          Your image opens preloaded in the{' '}
          <button className="font-semibold text-cyan-300 hover:underline" onClick={() => onNavigate('analyze')}>
            Analyze page
          </button>
          .
        </p>
      </section>

      {/* 4-step pipeline */}
      <section className="mt-12">
        <div className="grid gap-3 sm:grid-cols-4">
          {['Upload Image', 'U-Net Segmentation', 'Land-Cover Analysis', 'Site Suitability'].map(
            (s, i) => (
              <div key={s} className="glass-card flex items-center gap-3 p-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 font-bold text-emerald-300 ring-1 ring-emerald-400/25">
                  {i + 1}
                </span>
                <span className="text-sm font-semibold text-slate-100">{s}</span>
              </div>
            ),
          )}
        </div>
      </section>

      {/* Class strip */}
      <section className="glass-card mt-8 flex flex-wrap items-center gap-x-4 gap-y-2 p-4">
        <span className="text-xs font-bold uppercase tracking-widest text-slate-500">
          7 land-cover classes
        </span>
        {LAND_COVER_CLASSES.map((c) => (
          <span key={c.id} className="flex items-center gap-1.5 text-xs text-slate-300">
            <span className="h-3 w-3 rounded-[3px]" style={{ backgroundColor: c.color }} />
            {c.name}
          </span>
        ))}
      </section>

      {/* Project info */}
      <section className="mt-8 grid gap-3 sm:grid-cols-4">
        {[
          { k: 'Dataset', v: 'DeepGlobe Land Cover' },
          { k: 'Architecture', v: 'U-Net' },
          { k: 'Backend', v: 'Python Flask' },
          { k: 'Framework', v: 'TensorFlow / Keras' },
        ].map((it) => (
          <div key={it.k} className="glass-card p-4">
            <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500">{it.k}</div>
            <div className="mt-1 text-sm font-semibold text-slate-100">{it.v}</div>
          </div>
        ))}
      </section>
    </div>
  );
}
