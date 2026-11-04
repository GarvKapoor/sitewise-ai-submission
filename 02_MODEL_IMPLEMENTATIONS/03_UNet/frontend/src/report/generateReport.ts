import { jsPDF } from 'jspdf';
import type { PredictionResponse } from '../types';

/**
 * Export an Analysis Report PDF: original image, segmentation visualisation,
 * land-cover distribution, reliability and suitability rankings.
 * Demonstration results are prominently labelled in the exported file.
 */
export function generateReport(result: PredictionResponse): void {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const W = doc.internal.pageSize.getWidth();
  const M = 14; // margin
  let y = 18;

  // Per-section provenance — each model is labelled with its own status,
  // never a single global real/demo switch.
  const segReal = result.sections_status?.segmentation === 'connected';
  const segFailed = result.sections_status?.segmentation === 'failed';
  const relCnn = result.reliability?.model_used === true;
  const reliabilityFailed = result.sections_status?.reliability === 'failed';
  const demo = !segReal; // banner reflects segmentation; reliability labelled separately below

  // ---- Header
  doc.setFillColor(10, 17, 34);
  doc.rect(0, 0, W, 26, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.text('SiteWise AI — U-Net Analysis Report', M, 12);
  doc.setFontSize(9);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(150, 160, 180);
  doc.text(
    `${result.image.filename} · ${result.image.width}×${result.image.height} px · ${new Date().toLocaleString()}`,
    M,
    19,
  );
  if (segFailed) {
    doc.setFillColor(248, 113, 113);
    doc.roundedRect(W - 62, 8, 48, 9, 2, 2, 'F');
    doc.setTextColor(30, 5, 5);
    doc.setFont('helvetica', 'bold');
    doc.text('SEGMENTATION FAILED', W - 58, 14);
    doc.setTextColor(150, 160, 180);
    doc.setFont('helvetica', 'normal');
  } else if (demo) {
    doc.setFillColor(245, 158, 11);
    doc.roundedRect(W - 62, 8, 48, 9, 2, 2, 'F');
    doc.setTextColor(30, 20, 0);
    doc.setFont('helvetica', 'bold');
    doc.text('DEMONSTRATION DATA', W - 58, 14);
    doc.setTextColor(150, 160, 180);
    doc.setFont('helvetica', 'normal');
  } else {
    doc.setFillColor(52, 211, 153);
    doc.roundedRect(W - 62, 8, 48, 9, 2, 2, 'F');
    doc.setTextColor(4, 40, 28);
    doc.setFont('helvetica', 'bold');
    doc.text('ACTUAL U-NET OUTPUT', W - 58, 14);
    doc.setTextColor(150, 160, 180);
    doc.setFont('helvetica', 'normal');
  }
  y = 36;

  const sectionTitle = (t: string) => {
    if (y > 260) {
      doc.addPage();
      y = 20;
    }
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(12);
    doc.setTextColor(20, 40, 60);
    doc.text(t, M, y);
    y += 4;
    doc.setDrawColor(220, 225, 232);
    doc.line(M, y, W - M, y);
    y += 6;
  };

  const body = (t: string) => {
    if (y > 265) {
      doc.addPage();
      y = 20;
    }
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(60, 70, 85);
    const lines = doc.splitTextToSize(t, W - 2 * M);
    doc.text(lines, M, y);
    y += lines.length * 5 + 3;
  };

  const addImg = (dataUrl: string | undefined, w: number, h: number) => {
    if (!dataUrl || y + h > 275) return;
    doc.addImage(dataUrl, 'PNG', M, y, w, h);
    y += h + 6;
  };

  // ---- Provenance notice
  if (segFailed) {
    doc.setFillColor(254, 226, 226);
    const lines = doc.splitTextToSize(
      'The connected U-Net failed at inference for this image, so no segmentation output is included. Demonstration heuristics were deliberately NOT substituted — the other sections below still show their own labelled provenance.',
      W - 2 * M,
    );
    const h = lines.length * 5 + 8;
    doc.roundedRect(M, y, W - 2 * M, h, 2, 2, 'F');
    doc.setTextColor(153, 27, 27);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9.5);
    doc.text(lines, M + 4, y + 7);
    y += h + 8;
  } else if (demo) {
    doc.setFillColor(255, 243, 219);
    const lines = doc.splitTextToSize(
      'Demonstration mode: no trained U-Net was connected, so the segmentation mask comes from deterministic colour heuristics and is NOT a real model prediction. Reliability and suitability are labelled with their own provenance below.',
      W - 2 * M,
    );
    const h = lines.length * 5 + 8;
    doc.roundedRect(M, y, W - 2 * M, h, 2, 2, 'F');
    doc.setTextColor(154, 92, 0);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9.5);
    doc.text(lines, M + 4, y + 7);
    y += h + 8;
  }

  // ---- Original image
  sectionTitle('Original Satellite Image');
  const imgW = 110;
  const aspect = result.image.height / result.image.width;
  addImg(result.image.original_base64, imgW, imgW * aspect);

  // ---- Segmentation visualisation
  if (result.segmentation) {
    sectionTitle(
      segReal
        ? 'U-Net Segmentation (actual model output)'
        : 'Segmentation (demonstration heuristics — NOT a U-Net prediction)',
    );
    addImg(result.segmentation.mask_base64, imgW, imgW * aspect);
    addImg(result.segmentation.overlay_base64, imgW, imgW * aspect);
  } else {
    sectionTitle('U-Net Segmentation');
    body(
      'No segmentation output for this request' +
        (segFailed ? ' — the connected U-Net failed at inference.' : '.'),
    );
  }

  // ---- Land-cover distribution
  if (result.segmentation) {
    sectionTitle('Land-Cover Distribution');
    const rows = result.segmentation.class_percentages.filter((c) => c.percent > 0);
    rows.forEach((c) => {
      if (y > 270) {
        doc.addPage();
        y = 20;
      }
      const hex = c.color.replace('#', '');
      const rgb = [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
      doc.setFillColor(rgb[0], rgb[1], rgb[2]);
      doc.rect(M, y - 3, 5, 5, 'F');
      doc.setTextColor(60, 70, 85);
      doc.setFont('helvetica', 'normal');
      doc.text(`${c.name}`, M + 9, y + 0.5);
      doc.setFont('helvetica', 'bold');
      doc.text(`${c.percent.toFixed(2)}%  (${c.pixel_count.toLocaleString()} px)`, M + 70, y + 0.5);
      y += 7;
    });
    if (result.segmentation.dominant_class) {
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(20, 40, 60);
      doc.text(
        `Dominant class: ${result.segmentation.dominant_class.name} (${result.segmentation.dominant_class.percent.toFixed(2)}%)`,
        M,
        y + 2,
      );
      y += 8;
    }
  }

  // ---- Reliability (own provenance, independent of segmentation)
  sectionTitle('Image Reliability');
  if (result.reliability) {
    const provenance = reliabilityFailed
      ? 'reliability CNN inference FAILED — labelled heuristic fallback used (not a CNN score)'
      : relCnn
        ? 'assessed by the dedicated reliability CNN'
        : 'labelled heuristic fallback — reliability CNN not connected (NOT a CNN score)';
    body(
      `Score: ${result.reliability.score.toFixed(1)}% · status: ${result.reliability.status.toUpperCase()} · ` +
        `threshold: ${result.reliability.threshold}% · ${result.reliability.passed ? 'PASS' : 'FAIL'}. ` +
        `Provenance: ${provenance}.`,
    );
  } else {
    body('Unavailable.');
  }

  // ---- Visual features
  sectionTitle('Visual Features (RGB)');
  result.visual_features.forEach((f) => {
    if (y > 270) {
      doc.addPage();
      y = 20;
    }
    doc.text(`• ${f.name}: ${f.display}${f.unit}`, M + 2, y);
    y += 5.5;
  });
  y += 2;

  // ---- Suitability rankings (always deterministic engine, never an NN)
  sectionTitle('Site Suitability Screening (preliminary, deterministic scoring engine)');
  body(
    'Ranking produced by the deterministic screening engine (project-defined rules, identical inputs → identical scores). These are screening scores, not probabilities.',
  );
  [...result.suitability]
    .sort((a, b) => b.score - a.score)
    .forEach((s) => {
      if (y > 270) {
        doc.addPage();
        y = 20;
      }
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(20, 40, 60);
      doc.text(`${s.rank}. ${s.category} — ${s.score.toFixed(1)}${s.is_top ? '  (top pick)' : ''}`, M + 2, y);
      y += 5;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9);
      doc.setTextColor(90, 100, 115);
      const lines = doc.splitTextToSize(s.explanation, W - 2 * M - 6);
      doc.text(lines, M + 6, y);
      y += lines.length * 4.2 + 2.5;
      doc.setFontSize(10);
    });

  // ---- Disclaimer
  if (y > 250) {
    doc.addPage();
    y = 20;
  }
  doc.setFontSize(8.5);
  doc.setTextColor(120, 128, 140);
  const disc = doc.splitTextToSize(
    'Disclaimer: SiteWise AI provides preliminary, project-defined screening recommendations. It is not a construction approval or a legally verified land-use decision.',
    W - 2 * M,
  );
  doc.text(disc, M, y + 4);

  const stamp = demo ? 'sitewise-report-demo' : 'sitewise-report';
  doc.save(`${stamp}-${result.image.filename.replace(/\.[^.]+$/, '')}.pdf`);
}
