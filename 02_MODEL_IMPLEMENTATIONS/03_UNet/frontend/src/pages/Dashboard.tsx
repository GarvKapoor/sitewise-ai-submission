import { useEffect, useState } from 'react';
import type { ReliabilityHealth } from '../types';
import { fetchMetrics, type PerformanceMetricsResponse } from '../api/client';
import { RELIABILITY_THRESHOLD } from '../constants';

interface Props {
  health: ReliabilityHealth | null;
  healthLoading: boolean;
  healthError: string | null;
}

/**
 * Operations dashboard: backend/model status, DeepGlobe data facts,
 * a Model Performance section (real metrics or "Evaluation pending") and
 * model drop-in instructions.
 */
export default function Dashboard({ health, healthLoading, healthError }: Props) {
  const [metrics, setMetrics] = useState<PerformanceMetricsResponse | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    fetchMetrics()
      .then((m) => alive && setMetrics(m))
      .catch(() => alive && setMetrics(null))
      .finally(() => alive && setMetricsLoading(false));
    return () => {
      alive = false;
    };
  }, [health]);

  const getInfo = () => {
    if (healthLoading) return 'Checking backend…';
    if (!health) return `Unreachable${healthError ? ` — ${healthError}` : ''}`;
    return health.mode === 'real'
      ? 'Connected · real model inference'
      : 'Connected · demonstration mode';
  };

  const unet = health?.models.unet_connected ?? false;
  const rel = health?.models.reliability_connected ?? false;

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <header className="mb-8">
        <h1 className="text-3xl font-extrabold text-white sm:text-4xl">
          Operations <span className="accent-gradient-text">Dashboard</span>
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Backend status, models, dataset facts and evaluation results.
        </p>
      </header>

      <div className="grid gap-4 md:grid-cols-3">
        <StatusCard title="Flask backend" value={getInfo()} ok={!!health} detail="GET /api/health" />
        <StatusCard
          title="U-Net model"
          value={unet ? 'Loaded' : 'Not connected'}
          ok={unet}
          detail={health ? `${health.segmentation.input_size[0]}×${health.segmentation.input_size[1]} · ${health.segmentation.num_classes} classes` : 'awaiting backend'}
        />
        <StatusCard
          title="Reliability CNN"
          value={rel ? 'Loaded' : 'Not connected'}
          ok={rel}
          detail={`threshold ${health?.reliability.threshold ?? RELIABILITY_THRESHOLD}%`}
        />
      </div>

      {/* Model Performance */}
      <section className="glass-card mt-6 p-6">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-lg font-bold text-white">Model Performance</h2>
          {metrics?.available ? (
            <span className="chip text-[10px] border-emerald-500/40 bg-emerald-500/10 text-emerald-300">
              Real evaluation results
            </span>
          ) : (
            <span className="chip text-[10px] border-amber-500/40 bg-amber-500/10 text-amber-300">
              Evaluation pending
            </span>
          )}
        </div>
        {metricsLoading ? (
          <p className="mt-4 text-sm text-slate-400">Loading evaluation data…</p>
        ) : metrics?.available && metrics.unet ? (
          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            {[
              { k: 'Training Accuracy', v: `${(metrics.unet.training_accuracy * 100).toFixed(1)}%` },
              { k: 'Validation Loss', v: metrics.unet.validation_loss.toFixed(3) },
              { k: 'Mean IoU', v: metrics.unet.mean_iou.toFixed(3) },
              { k: 'Dice Coefficient', v: metrics.unet.dice.toFixed(3) },
            ].map((m) => (
              <div key={m.k} className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 p-4 text-center">
                <div className="text-xl font-extrabold text-white">{m.v}</div>
                <div className="mt-1 text-[11px] font-medium text-slate-400">{m.k}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-4 text-sm text-slate-400">
            {metrics?.message ??
              'Evaluation pending — metrics appear only after real training and evaluation.'}
          </p>
        )}
      </section>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <section className="glass-card p-6">
          <h2 className="text-lg font-bold text-white">Segmentation configuration</h2>
          <dl className="mt-4 space-y-3 text-sm">
            <Row k="Classes" v={health ? String(health.segmentation.num_classes) : '7'} />
            <Row k="Class names" v={health ? health.segmentation.class_names.join(', ') : 'Urban, Agriculture, Rangeland, Forest, Water, Barren, Unknown'} />
            <Row k="Dataset" v={health?.segmentation.dataset ?? 'DeepGlobe Land Cover'} />
            <Row k="Input size" v={health ? `${health.segmentation.input_size[0]} × ${health.segmentation.input_size[1]} px` : '—'} />
            <Row k="Model formats" v=".keras · .h5" />
          </dl>
        </section>

        <section className="glass-card p-6">
          <h2 className="text-lg font-bold text-white">Data facts</h2>
          <ul className="mt-4 space-y-3 text-sm text-slate-400">
            <li>🛰️ <span className="font-semibold text-slate-200">DeepGlobe Land Cover</span> — RGB satellite tiles with ground-truth pixel-level masks (7 land-cover classes).</li>
            <li>🗺️ Masks define what counts as correct; class colours are display-constants shared across legends, charts and masks.</li>
            <li>📊 No accuracy, IoU or Dice values are shown unless the real evaluation file exists — the site never fabricates metrics.</li>
          </ul>
        </section>
      </div>

      <div className="glass-card mt-6 p-6">
        <h2 className="text-lg font-bold text-white">Connect your trained models</h2>
        <ol className="mt-4 space-y-2 text-sm text-slate-400">
          <li>1. Train the U-Net offline (kept separate from the website).</li>
          <li>2. Drop <code className="rounded bg-navy-850 px-1 text-cyan-300">.keras</code> / <code className="rounded bg-navy-850 px-1 text-cyan-300">.h5</code> files into <code className="rounded bg-navy-850 px-1 text-cyan-300">backend/models/</code>.</li>
          <li>3. Match class order/size in <code className="rounded bg-navy-850 px-1 text-cyan-300">backend/config/model_config.py</code>; optionally add <code className="rounded bg-navy-850 px-1 text-cyan-300">models/evaluation.json</code> for the performance section.</li>
          <li>4. Restart the backend — <code className="rounded bg-navy-850 px-1 text-cyan-300">GET /api/health</code> flips the site from DEMONSTRATION DATA to real output.</li>
        </ol>
      </div>
    </div>
  );
}

function StatusCard({ title, value, ok, detail }: { title: string; value: string; ok: boolean; detail: string }) {
  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">{title}</h3>
        <span className={`chip text-[10px] ${ok ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/40 bg-amber-500/10 text-amber-300'}`}>
          {ok ? 'OK' : 'Pending'}
        </span>
      </div>
      <div className="mt-2 text-xl font-extrabold text-white">{value}</div>
      <div className="mt-1 text-xs text-slate-500">{detail}</div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex flex-wrap justify-between gap-2 border-b border-white/5 pb-2">
      <dt className="text-slate-500">{k}</dt>
      <dd className="text-right font-medium text-slate-200">{v}</dd>
    </div>
  );
}
