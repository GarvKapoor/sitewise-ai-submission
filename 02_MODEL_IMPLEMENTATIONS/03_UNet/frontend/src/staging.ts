import type { UploadState } from './components/UploadPanel';

/**
 * Hand-off of an upload selected on the landing page to the Analyze page,
 * so the landing can offer a prominent upload section without duplicating
 * the analysis flow.
 */
let pending: UploadState | null = null;

export function setPendingUpload(u: UploadState | null) {
  pending = u;
}

export function consumePendingUpload(): UploadState | null {
  const p = pending;
  pending = null;
  return p;
}
