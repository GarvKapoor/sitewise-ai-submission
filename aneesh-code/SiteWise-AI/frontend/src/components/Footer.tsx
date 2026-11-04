import type { Route } from '../App';

export default function Footer({ onNavigate }: { onNavigate: (to: Route) => void }) {
  return (
    <footer className="border-t border-white/10 bg-navy-950/60">
      <div className="mx-auto grid max-w-7xl gap-8 px-4 py-10 sm:px-6 md:grid-cols-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500">
              <svg viewBox="0 0 24 24" className="h-4 w-4 text-navy-950" fill="none" stroke="currentColor" strokeWidth="2.4">
                <path d="M4 18l4-8 3.5 5L14 11l6 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
            <span className="font-extrabold tracking-tight text-white">
              SiteWise <span className="accent-gradient-text">AI</span>
            </span>
          </div>
          <p className="mt-3 max-w-xs text-sm text-slate-400">
            AI-powered satellite image segmentation and preliminary site suitability analysis
            powered by a U-Net semantic segmentation architecture.
          </p>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-white">Explore</h4>
          <ul className="mt-3 space-y-2 text-sm text-slate-400">
            <li><button className="hover:text-emerald-300" onClick={() => onNavigate('analyze')}>Analyze Image</button></li>
            <li><button className="hover:text-emerald-300" onClick={() => onNavigate('how')}>How It Works</button></li>
            <li><button className="hover:text-emerald-300" onClick={() => onNavigate('about')}>About</button></li>
            <li><button className="hover:text-emerald-300" onClick={() => onNavigate('dashboard')}>Dashboard</button></li>
          </ul>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-white">Disclaimer</h4>
          <p className="mt-3 text-sm text-slate-400">
            SiteWise AI provides project-defined <span className="text-slate-200">preliminary site-use
            screening recommendations</span>. It is not a construction approval, engineering
            feasibility study, or legally verified land-use decision.
          </p>
        </div>
      </div>
      <div className="border-t border-white/5 py-4 text-center text-xs text-slate-500">
        © {new Date().getFullYear()} SiteWise AI · U-Net semantic segmentation · DeepGlobe Land Cover
      </div>
    </footer>
  );
}
