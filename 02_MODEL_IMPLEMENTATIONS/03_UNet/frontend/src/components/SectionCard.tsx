import type { ReactNode } from 'react';

interface Props {
  id?: string;
  letter?: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}

/** Glassmorphism section card used across the results dashboard (Sections A–E). */
export default function SectionCard({ id, letter, title, subtitle, children, className = '' }: Props) {
  return (
    <section id={id} className={`glass-card p-5 sm:p-7 ${className}`}>
      <div className="mb-5 flex items-start gap-3">
        {letter && (
          <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 font-bold text-emerald-300 ring-1 ring-emerald-400/30">
            {letter}
          </span>
        )}
        <div>
          <h2 className="text-xl font-bold text-white sm:text-2xl">{title}</h2>
          {subtitle && <p className="mt-1 text-sm text-slate-400">{subtitle}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}
