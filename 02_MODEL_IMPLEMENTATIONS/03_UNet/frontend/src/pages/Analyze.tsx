import { useCallback, useRef, useState } from 'react';
import UploadPanel, { type UploadState } from '../components/UploadPanel';
import ModelStatusBanner from '../components/ModelStatusBanner';
import AnalysisProgress, { PIPELINE_STEPS } from '../components/AnalysisProgress';
import SectionCard from '../components/SectionCard';
import SegmentationViewer from '../components/SegmentationViewer';
import ClassLegend from '../components/ClassLegend';
import ClassDistribution from '../components/ClassDistribution';
import ReliabilityCard from '../components/ReliabilityCard';
import FeatureCards from '../components/FeatureCards';
import SuitabilityBars from '../components/SuitabilityBars';
import ModelInfo from '../components/ModelInfo';
import { DemoBadge, RealBadge } from '../components/DemoBadge';
import { predictImage } from '../api/client';
import { consumePendingUpload } from '../staging';
import { generateReport } from '../report/generateReport';
import type { PredictionResponse, ReliabilityHealth } from '../types';

interface Props {
  health: ReliabilityHealth | null;
  healthLoading: boolean;
  healthError: string | null;
  onRefreshHealth: () => Promise<void>;
}

export default function Analyze({ health, healthLoading, healthError, onRefreshHealth }: Props) {
  const [upload, setUpload] = useState<UploadState | null>(() => consumePendingUpload());
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [progressStep, setProgressStep] = useState(0);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const runAnalysis = useCallback(async () => {
    if (!upload || busy) return;
    setBusy(true);
    setAnalysisError(null);
    setResults(null);
    setProgressStep(0);
    const ticker = window.setInterval(() => {
      setProgressStep((s) => Math.min(s + 1, PIPELINE_STEPS.length - 2));
    }, 900);

    try {
      const res = await predictImage(upload.file);
      window.clearInterval(ticker);
      setResults(res);
      setProgressStep(PIPELINE_STEPS.length);
      window.setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth' }), 120);
    } catch (e) {
      window.clearInterval(ticker);
      setAnalysisError(
        e instanceof Error ? e.message : 'Analysis failed — the backend could not process this image.',
      );
    } finally {
      setBusy(false);
    }
  }, [upload, busy]);

  const removeUpload = useCallback(() => {
    if (upload) URL.revokeObjectURL(upload.previewUrl);
    setUpload(null);
    setResults(null);
    setAnalysisError(null);
  }, [upload]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <header className="mb-6">
        <h1 className="text-3xl font-extrabold text-white sm:text-4xl">
          Analyze <span className="accent-gradient-text">Satellite Image</span>
        </h1>
        <p className="mt-2 text-sm text-slate-400">
          Upload → U-Net segmentation → land-cover analysis → site suitability.
        </p>
      </header>

      <div className="mb-6">
        <ModelStatusBanner
          health={health}
          healthLoading={healthLoading}
          healthError={healthError}
          onRefresh={() => void onRefreshHealth()}
        />
      </div>

      {/* Upload */}
      <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
        <UploadPanel
          upload={upload}
          busy={busy}
          onSelect={(u) => setUpload(u)}
          onRemove={removeUpload}
        />
        <div className="glass-card flex flex-col justify-between gap-4 p-5">
          <div>
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
              4-Step Pipeline
            </h3>
            <ol className="mt-3 space-y-2 text-xs text-slate-400">
              {PIPELINE_STEPS.map((s, i) => (
                <li key={s.key} className="flex gap-2.5">
                  <span className="font-mono text-cyan-400">{i + 1}</span>
                  {s.label}
                </li>
              ))}
            </ol>
          </div>
          <button
            className="btn-primary w-full"
            disabled={!upload || busy}
            onClick={() => void runAnalysis()}
          >
            {busy && (
              <span className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-navy-950/30 border-t-navy-950" />
            )}
            {busy ? 'Analyzing…' : 'Analyze Satellite Image'}
          </button>
          {!upload && <p className="text-center text-xs text-slate-500">Upload an image to begin</p>}
        </div>
      </div>

      {busy && (
        <div className="mt-8">
          <AnalysisProgress currentStep={progressStep} />
        </div>
      )}

      {analysisError && (
        <div className="mt-8 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
          {analysisError}
        </div>
      )}

      {/* Results dashboard */}
      {results && (
        <div ref={resultsRef} className="mt-10 scroll-mt-24 space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="text-2xl font-extrabold text-white">Results</h2>
            {/* Per-model provenance (not one global demo flag): */}
            {results.sections_status.segmentation === 'connected' ? <RealBadge /> : <DemoBadge />}
            {results.sections_status.segmentation === 'failed' && (
              <span className="chip text-[10px] border-red-500/40 bg-red-500/10 font-bold uppercase tracking-wider text-red-300">
                U-Net inference failed
              </span>
            )}
            {results.sections_status.reliability === 'connected' ? (
              <span className="chip text-[10px] border-emerald-500/40 bg-emerald-500/10 font-bold uppercase tracking-wider text-emerald-300">
                Reliability CNN assessed
              </span>
            ) : results.sections_status.reliability === 'failed' ? (
              <span className="chip text-[10px] border-red-500/40 bg-red-500/10 font-bold uppercase tracking-wider text-red-300">
                Reliability CNN error · heuristic fallback used
              </span>
            ) : (
              <span className="chip text-[10px] border-amber-500/40 bg-amber-500/10 font-bold uppercase tracking-wider text-amber-300">
                Reliability CNN unavailable · heuristic fallback used
              </span>
            )}
            <button
              className="ml-auto btn-secondary !px-4 !py-2 text-sm"
              onClick={() => generateReport(results)}
              title="Export a one-page PDF of this analysis"
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3v12m0 0 4-4m-4 4-4-4M4 19h16" />
              </svg>
              Download Analysis Report
            </button>
          </div>

          {/* Main focus — segmentation visualisation */}
          <SectionCard
            title="Segmentation Visualisation"
            subtitle={
              results.sections_status.segmentation === 'connected'
                ? 'Actual U-Net output — original image, mask, overlay, before/after.'
                : 'No real segmentation output for this request.'
            }
          >
            <SegmentationViewer result={results} />
            <div className="mt-5">
              <ClassLegend />
            </div>
            {results.segmentation && (
              <div className="mt-5 rounded-xl border border-white/10 bg-navy-850/60 p-4">
                <ClassDistribution result={results} />
              </div>
            )}
          </SectionCard>

          {/* Compact summary */}
          <div className="grid gap-4 lg:grid-cols-[repeat(3,1fr)_minmax(260px,0.9fr)]">
            <SummaryTile
              label="Dominant land cover"
              value={results.segmentation?.dominant_class?.name ?? '—'}
              accent={results.segmentation?.dominant_class?.color}
            />
            <SummaryTile
              label="Segmentation confidence"
              value={results.segmentation?.confidence ? `${results.segmentation.confidence.mean.toFixed(1)}%` : 'n/a'}
            />
            <SummaryTile
              label="Processing"
              value={`${results.preprocessing.target_size[0]} × ${results.preprocessing.target_size[1]} px · analysed`}
            />
            <ReliabilityCard result={results} />
          </div>

          {/* Features */}
          <SectionCard title="Visual Features" subtitle="Eight RGB-based metric cards.">
            <FeatureCards features={results.visual_features} />
          </SectionCard>

          {/* Suitability */}
          <SectionCard title="Site Suitability" subtitle="Eight ranked preliminary screening scores (0–100, not probabilities).">
            <SuitabilityBars result={results} />
          </SectionCard>

          {/* Model info */}
          <SectionCard title="Models Used">
            <ModelInfo health={health} />
          </SectionCard>

          {results.warnings.length > 0 && (
            <ul className="space-y-1 rounded-xl border border-amber-500/25 bg-amber-500/5 p-4 text-xs text-amber-200">
              {results.warnings.map((w, i) => (
                <li key={i}>• {w}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function SummaryTile({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-navy-850/60 p-4">
      <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-2">
        {accent && (
          <span className="h-3.5 w-3.5 rounded-[4px] ring-1 ring-white/25" style={{ backgroundColor: accent }} />
        )}
        <span className="text-lg font-extrabold text-white">{value}</span>
      </div>
    </div>
  );
}
