"""
Site suitability screening engine (Model 3 — deterministic scoring engine,
NOT a trained neural network).

Input — 15 available values in total:
  • 7 DeepGlobe land-cover proportions (%) from the U-Net segmentation:
      Urban, Agriculture, Rangeland, Forest, Water, Barren, Unknown
  • 8 RGB-based visual features (0-1) from services/preprocessing.py:
      mean_red, mean_green, mean_blue, avg_brightness,
      vegetation_ratio, water_ratio, builtup_ratio, road_indicator
      (the road_indicator is an RGB-based proxy — share of bright,
      low-saturation pixels — NOT real road segmentation, because the
      DeepGlobe class set has no separate road class)

Feature usage — NOT all 15 inputs feed every score:
  • Land-cover proportions drive every category through the positives /
    negatives weights below (this is the main signal).
  • vegetation_ratio (vegetation_ratio feature) additionally contributes to:
      Green / Conservation, Agricultural Use, and (small nudges) Educational,
      Healthcare and Commercial Development.
  • builtup_ratio contributes to every "* Development" category
      (Residential, Commercial, Industrial, Educational, Healthcare).
  • road_indicator contributes only to Transportation Infrastructure.
  • water_ratio contributes a small negative nudge to Educational,
      Healthcare and Commercial Development.
  • The four intensity features (mean red / green / blue, brightness) are
    reported on the dashboard for transparency but are NOT used in scoring.

Output: eight project-defined suitability scores (0-100) with short
explanations. These are *preliminary screening* scores — not probabilities,
not approvals, and not legal or engineering feasibility verdicts.
Scores are fully deterministic: identical inputs always produce identical
outcomes. Weights are project-defined heuristics, documented (not learned).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Category definitions. Each category blanks a base score plus positive /
# negative factor references. Weights are project-defined screening heuristics
# — tune them with project stakeholders; they are clearly documented, not
# learned from data.
# ---------------------------------------------------------------------------
CATEGORIES: List[Dict[str, Any]] = [
    {
        "name": "Residential Development",
        "base": 30,
        "positives": [("Urban", 1.3), ("Barren", 0.25), ("Rangeland", 0.3)],
        "negatives": [("Water", -0.7), ("Forest", -0.25), ("Barren", -0.1)],
        "cap": 92,
    },
    {
        "name": "Commercial Development",
        "base": 25,
        "positives": [("Urban", 0.9), ("Barren", 0.3)],
        "negatives": [("Forest", -0.4), ("Water", -0.3), ("Agriculture", -0.2)],
        "cap": 90,
    },
    {
        "name": "Industrial Development",
        "base": 22,
        "positives": [("Urban", 1.7), ("Barren", 0.7)],
        "negatives": [("Agriculture", -0.4), ("Water", -0.3), ("Forest", -0.35), ("Rangeland", -0.2)],
        "cap": 90,
    },
    {
        "name": "Educational Development",
        "base": 28,
        "positives": [("Urban", 1.0), ("Rangeland", 0.2), ("Barren", 0.3)],
        "negatives": [("Water", -0.3)],
        "cap": 88,
    },
    {
        "name": "Healthcare Development",
        "base": 26,
        "positives": [("Urban", 1.2), ("Rangeland", 0.2)],
        "negatives": [("Water", -0.3)],
        "cap": 88,
    },
    {
        "name": "Agricultural Use",
        "base": 30,
        "positives": [("Agriculture", 1.7), ("Rangeland", 0.5)],
        "negatives": [("Urban", -0.5), ("Water", -0.2)],
        "cap": 95,
    },
    {
        "name": "Transportation Infrastructure",
        "base": 24,
        "positives": [("Urban", 1.5), ("Barren", 0.5)],
        "negatives": [("Water", -0.5), ("Forest", -0.3), ("Agriculture", -0.2)],
        "cap": 92,
    },
    {
        "name": "Green / Conservation",
        "base": 25,
        "positives": [
            ("Forest", 1.7),
            ("Rangeland", 1.2),
            ("Water", 0.4),
            ("Agriculture", 0.3),
        ],
        "negatives": [("Urban", -0.9), ("Barren", -0.4)],
        "cap": 97,
    },
]

VEGETATION_FEATURES = ("vegetation_ratio",)
WATER_FEATURES = ("water_ratio",)
BUILT_FEATURES = ("builtup_ratio", "road_indicator")


def compute_scores(
    class_percentages: Optional[List[Dict[str, Any]]],
    visual_features: List[Dict[str, Any]],
    name_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    proportions = {c["name"]: float(c["percent"]) / 100.0 for c in (class_percentages or [])}
    features = {f["key"]: float(f["fraction"]) for f in visual_features}

    results: List[Dict[str, Any]] = []
    for cat in CATEGORIES:
        score = float(cat["base"])
        factors: List[str] = []

        def pct(name: str) -> float:
            return proportions.get(name, 0.0)

        def feat(key: str) -> float:
            return features.get(key, 0.0)

        for name, weight in cat["positives"]:
            v = pct(name)
            if v > 0.01:
                score += weight * v * 60.0
                factors.append(f"{name.title()} {v * 100:.1f}% · +")

        for name, weight in cat["negatives"]:
            v = pct(name)
            if v > 0.01:
                score += weight * v * 60.0
                factors.append(f"{name.title()} {v * 100:.1f}% · −")

        # Visual-feature nudges (bounded, small relative to class proportions)
        veg = feat("vegetation_ratio")
        wat = feat("water_ratio")
        bui = feat("builtup_ratio")
        road = feat("road_indicator")
        if cat["name"].startswith("Green"):
            score += 14.0 * veg
            factors.append(f"Vegetation {veg * 100:.0f}% · +")
        if cat["name"] == "Transportation Infrastructure":
            score += 16.0 * road
            factors.append(f"Road indicator {road * 100:.0f}% · +")
        if "Development" in cat["name"]:
            score += 12.0 * bui
            if bui > 0.02:
                factors.append(f"Built-up {bui * 100:.0f}% · +")
        if cat["name"] in ("Agricultural Use", "Green / Conservation"):
            score += 8.0 * veg
        if cat["name"] in ("Educational Development", "Healthcare Development", "Commercial Development"):
            score += 5.0 * veg
            score -= 6.0 * wat

        score = max(0.0, min(float(cat["cap"]), score))

        # Deterministic explanation text
        top = max(proportions.items(), key=lambda kv: kv[1], default=("", 0.0))
        explanation = (
            f"Screened from {top[0].lower()} dominance ({top[1] * 100:.1f}%)"
            if top[0]
            else "Screened from the 15-value feature representation"
        )
        if cat["name"].startswith("Green") and veg > 0.05:
            explanation += "; vegetation presence supports conservation screening"
        if bui > 0.05 and "Development" in cat["name"]:
            explanation += "; prior built-up presence supports this category"

        results.append(
            {
                "category": cat["name"],
                "score": round(score, 1),
                "rank": 0,
                "explanation": explanation,
                "contributing_factors": factors[:6],
                "is_top": False,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    results[0]["is_top"] = True
    results[0]["explanation"] = f"Highest screened category. {results[0]['explanation']}"
    return results
