/**
 * DeepGlobe Land Cover — standardised 7-class mapping used across the whole
 * website (backend legends, masks, charts and these constants match).
 */
export interface LandCoverClass {
  id: number;
  name: string;
  color: string;
}

export const LAND_COVER_CLASSES: LandCoverClass[] = [
  { id: 0, name: 'Urban', color: '#ef4444' },
  { id: 1, name: 'Agriculture', color: '#84cc16' },
  { id: 2, name: 'Rangeland', color: '#eab308' },
  { id: 3, name: 'Forest', color: '#16a34a' },
  { id: 4, name: 'Water', color: '#3b82f6' },
  { id: 5, name: 'Barren', color: '#d6c39b' },
  { id: 6, name: 'Unknown', color: '#6b7280' },
];

export const classById = (id: number): LandCoverClass | undefined =>
  LAND_COVER_CLASSES.find((c) => c.id === id);

export const RELIABILITY_THRESHOLD = 55;

export const DISCLAIMER =
  'Preliminary project-defined screening — not a construction approval or legally verified land-use decision.';
