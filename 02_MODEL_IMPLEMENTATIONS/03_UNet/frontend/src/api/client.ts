import type {
  BackendStatus,
  PredictionResponse,
  ReliabilityHealth,
} from '../types';

/**
 * All requests go through the single '/api' prefix.
 *
 * Local development: Vite's dev server (frontend/vite.config.ts) proxies
 * EVERY '/api/*' request to the Flask backend at http://127.0.0.1:5000 and
 * rewrites the prefix away, so the browser sees:
 *   /api/health   -> Flask GET  /health
 *   /api/predict  -> Flask POST /predict
 *   /api/metrics  -> Flask GET  /metrics
 *
 * Production: put the static frontend behind a reverse proxy that routes
 * '/api/*' to the Flask service with the SAME rewrite (strip '/api'), e.g.
 * an NGINX location block:
 *   location /api/ {
 *     proxy_pass         http://flask-backend:5000/;
 *     proxy_set_header   Host $host;
 *     proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
 *   }
 * or set the Flask app behind the same domain under an alias that maps
 * '/api/health' -> '/health' etc. (see README "Production deployment").
 */
const API_BASE = '/api';

export interface PerformanceMetricsResponse {
  available: boolean;
  message?: string;
  source?: string;
  unet?: {
    training_accuracy: number;
    validation_loss: number;
    mean_iou: number;
    dice: number;
  };
}

async function handleJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) detail = body.error;
    } catch {
      /* keep statusText */
    }
    throw new Error(`Backend request failed (${res.status}): ${detail}`);
  }
  return (await res.json()) as T;
}

/**
 * GET /metrics — evaluation results appear only after real offline training/
 * evaluation (backend/models/evaluation.json). Otherwise: "Evaluation pending".
 */
export async function fetchMetrics(signal?: AbortSignal): Promise<PerformanceMetricsResponse> {
  const res = await fetch(`${API_BASE}/metrics`, { signal });
  return handleJson<PerformanceMetricsResponse>(res);
}

/**
 * GET /health — determines whether the Flask backend and the trained
 * U-Net / reliability models are connected.
 */
export async function fetchHealth(signal?: AbortSignal): Promise<ReliabilityHealth> {
  const res = await fetch(`${API_BASE}/health`, { signal });
  return handleJson<BackendStatus & ReliabilityHealth>(res);
}

/**
 * POST /predict — uploads a satellite image for preprocessing, U-Net
 * segmentation, reliability assessment and suitability scoring.
 */
export async function predictImage(file: File, signal?: AbortSignal): Promise<PredictionResponse> {
  const form = new FormData();
  form.append('image', file);
  const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form, signal });
  return handleJson<PredictionResponse>(res);
}
