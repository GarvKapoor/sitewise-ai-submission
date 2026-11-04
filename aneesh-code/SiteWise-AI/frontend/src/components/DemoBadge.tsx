interface DemoBadgeProps {
  compact?: boolean;
}

/**
 * Explicit "DEMONSTRATION DATA" marker. Everything the backend returns while
 * the real U-Net / reliability models are not connected is clearly labelled
 * with this badge and never mixed with actual model output.
 */
export function DemoBadge({ compact = false }: DemoBadgeProps) {
  return (
    <span
      className={`chip animate-pulse-slow border-amber-500/50 bg-amber-500/15 font-extrabold uppercase tracking-wider text-amber-300 ${
        compact ? 'text-[10px]' : 'text-xs'
      }`}
      title="No trained U-Net model is connected — this output is clearly-labelled demonstration data, not a real model prediction."
    >
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
        <path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      </svg>
      Demonstration data
    </span>
  );
}

export function RealBadge() {
  return (
    <span className="chip border-emerald-500/50 bg-emerald-500/15 text-xs font-extrabold uppercase tracking-wider text-emerald-300">
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> Real model output
    </span>
  );
}
