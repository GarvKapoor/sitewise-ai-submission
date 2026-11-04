/**
 * Shared TypeScript contracts between the SiteWise AI frontend and the
 * Flask backend (backend/app.py). Field names must stay in sync with the
 * JSON produced by the backend's POST /predict and GET /health endpoints.
 */

export type ApiResponseMode = 'real' | 'demo' | 'unavailable';

export interface BackendStatus {
  ok: boolean;
  mode: ApiResponseMode;
  models: {
    unet_connected: boolean;
    reliability_connected: boolean;
  };
  segmentation: {
    input_size: [number, number];
    num_classes: number;
    class_names: string[];
    dataset: string;
    dataset_note: string;
  };
  reliability: {
    threshold: number;
  };
}

export interface ClassPercentage {
  class_id: number;
  name: string;
  color: string;
  percent: number;
  pixel_count: number;
}

export interface SegmentationResult {
  mask_base64: string;
  overlay_base64: string;
  /** "unet" (real model) or "demonstration_heuristic" (labelled demo data) */
  source?: string;
  input_size: [number, number];
  class_percentages: ClassPercentage[];
  dominant_class: { class_id: number; name: string; color: string; percent: number } | null;
  confidence: {
    /** Mean softmax probability of the assigned class, 0-100 */
    mean: number;
    min: number;
    max: number;
  } | null;
  total_pixels: number;
}

export interface ReliabilityResult {
  /** 0-100 reliability score from the reliability CNN (or labelled heuristic fallback) */
  score: number;
  status: 'clean' | 'degraded' | 'unavailable';
  /** True when this score came from the dedicated reliability CNN */
  model_used: boolean;
  /** Provenance: "reliability_cnn" or "heuristic_fallback" */
  assessed_by?: string;
  threshold: number;
  passed: boolean;
  /** Set only when a CONNECTED model failed at inference time */
  inference_error?: string;
}

/** Per-model provenance states returned by POST /predict (Task: per-model
 *  status instead of one global real/demo switch). */
export type ModelProvenance = 'connected' | 'unavailable' | 'failed' | 'demonstration';

export interface SectionStatuses {
  /** connected = genuine U-Net output · demonstration = labelled demo data ·
   *  failed = model connected but inference raised */
  segmentation: ModelProvenance;
  /** connected = CNN scored it · unavailable = CNN missing (heuristic used) ·
   *  failed = CNN connected but inference raised */
  reliability: ModelProvenance;
  /** Always deterministic_engine — a scoring engine, not an NN */
  suitability_engine: 'deterministic_engine';
}

export interface VisualFeature {
  key: string;
  name: string;
  value: number;
  /** 0-1 normalised fraction used for the visual indicator */
  fraction: number;
  display: string;
  unit: string;
}

export interface SuitabilityResult {
  category: string;
  /** Project-defined screening score 0-100 — NOT a probability */
  score: number;
  rank: number;
  explanation: string;
  contributing_factors: string[];
  is_top: boolean;
}

export interface PredictionResponse {
  mode: ApiResponseMode;
  model_status: {
    unet_connected: boolean;
    reliability_connected: boolean;
  };
  /** Per-model provenance (added by the audited backend) */
  sections_status: SectionStatuses;
  image: {
    filename: string;
    original_base64: string;
    width: number;
    height: number;
  };
  preprocessing: {
    target_size: [number, number];
    normalization: string;
    steps: string[];
  };
  segmentation: SegmentationResult | null;
  reliability: ReliabilityResult | null;
  visual_features: VisualFeature[];
  suitability: SuitabilityResult[];
  top_recommendation: {
    category: string;
    score: number;
  } | null;
  warnings: string[];
}

export interface ReliabilityHealth {
  ok: boolean;
  mode: ApiResponseMode;
  models: {
    unet_connected: boolean;
    reliability_connected: boolean;
  };
  /** per-model states: connected / unavailable / failed */
  model_status?: { unet: ModelProvenance; reliability: ModelProvenance };
  segmentation: BackendStatus['segmentation'];
  reliability: BackendStatus['reliability'];
  message?: string;
}
