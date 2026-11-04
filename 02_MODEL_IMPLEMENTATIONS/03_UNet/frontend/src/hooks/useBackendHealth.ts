import { useCallback, useEffect, useState } from 'react';
import { fetchHealth } from '../api/client';
import type { ReliabilityHealth } from '../types';

/**
 * Polls GET /health to determine whether the Flask backend (and its
 * trained U-Net / reliability models) are connected. When unreachable,
 * the UI shows the "Model Not Connected" state andExplicit demo labels.
 */
export function useBackendHealth() {
  const [health, setHealth] = useState<ReliabilityHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch (e) {
      setHealth(null);
      setError(e instanceof Error ? e.message : 'Backend unreachable');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { health, loading, error, refresh };
}
