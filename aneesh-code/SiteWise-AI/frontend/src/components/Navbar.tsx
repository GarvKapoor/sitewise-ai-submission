import type { Route } from '../App';
import type { ReliabilityHealth } from '../types';

const LINKS: { id: Route; label: string }[] = [
  { id: 'home', label: 'Home' },
  { id: 'analyze', label: 'Analyze Image' },
  { id: 'how', label: 'How It Works' },
  { id: 'about', label: 'About' },
  { id: 'dashboard', label: 'Dashboard' },
];

function statusChip(health: ReliabilityHealth | null) {
  if (!health) return null;
  if (health.mode === 'real' && health.models.unet_connected) {
    return (
      <span className="chip border-emerald-500/40 bg-emerald-500/10 text-emerald-300">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" /> U-Net connected
      </span>
    );
  }
  if (health.mode === 'demo') {
    return (
      <span className="chip border-amber-500/40 bg-amber-500/10 text-amber-300">
        <span className="h-1.5 w-1.5 rounded-full bg-amber-400" /> Demo mode
      </span>
    );
  }
  return (
    <span className="chip border-red-500/40 bg-red-500/10 text-red-300">
      <span className="h-1.5 w-1.5 rounded-full bg-red-400" /> Model not connected
    </span>
  );
}

interface Props {
  route: Route;
  onNavigate: (to: Route) => void;
  health: ReliabilityHealth | null;
  healthLoading: boolean;
}

export default function Navbar({ route, onNavigate, health, healthLoading }: Props) {
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-navy-900/80 backdrop-blur-lg">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <button
          className="flex items-center gap-3"
          onClick={() => onNavigate('home')}
          aria-label="SiteWise AI home"
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 shadow-lg shadow-cyan-500/20">
            <svg viewBox="0 0 24 24" className="h-5 w-5 text-navy-950" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 18l4-8 3.5 5L14 11l6 7" />
              <circle cx="18.5" cy="5.5" r="2" fill="currentColor" stroke="none" />
            </svg>
          </span>
          <span className="text-left">
            <span className="block text-base font-extrabold tracking-tight text-white">
              SiteWise <span className="accent-gradient-text">AI</span>
            </span>
            <span className="block text-[10px] font-medium uppercase tracking-widest text-slate-400">
              U-Net segmentation
            </span>
          </span>
        </button>

        <nav className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <button
              key={l.id}
              onClick={() => onNavigate(l.id)}
              className={`rounded-lg px-3.5 py-2 text-sm font-medium transition ${
                route === l.id
                  ? 'bg-white/10 text-white'
                  : 'text-slate-300 hover:bg-white/5 hover:text-white'
              }`}
            >
              {l.label}
            </button>
          ))}
        </nav>

        <div className="hidden items-center gap-2 lg:flex">{!healthLoading && statusChip(health)}</div>

        <div className="flex items-center md:hidden">
          <select
            value={route}
            onChange={(e) => onNavigate(e.target.value as Route)}
            className="rounded-lg border border-white/15 bg-navy-850 px-3 py-2 text-sm text-slate-200"
            aria-label="Navigation"
          >
            {LINKS.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </header>
  );
}
