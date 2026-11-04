import { LAND_COVER_CLASSES } from '../constants';

export function ClassLegend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2">
      {LAND_COVER_CLASSES.map((c) => (
        <span key={c.id} className="flex items-center gap-2 text-xs font-medium text-slate-300">
          <span
            className="h-3.5 w-3.5 rounded-[4px] ring-1 ring-white/20"
            style={{ backgroundColor: c.color }}
          />
          {c.name}
        </span>
      ))}
    </div>
  );
}

export default ClassLegend;
