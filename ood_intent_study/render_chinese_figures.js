const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const resultRoot = path.resolve(process.argv[2] || "runs/ood_intent_study_f32");
const fontFamily = "Microsoft YaHei, SimHei, sans-serif";
const palette = ["#2474A6", "#C63C3C", "#2D8A4E", "#7856A6", "#D9822B", "#159A9C", "#71717A"];

const readoutNames = {
  last: "最后位置",
  non_image_mean: "非图像位置平均",
  image_mean: "图像位置平均",
};

const attackNames = {
  "CS-DJ": "CS-DJ（复杂多图干扰）",
  FigStep: "FigStep（文字排版）",
  JOOD: "JOOD（图像混合）",
};

const modelNames = {
  qwen25vl7b: "Qwen2.5-VL-7B",
  gemma3_12b: "Gemma3-12B",
};

const panelNames = {
  all: "全部样本组",
  text_only: "纯文本组",
  multimodal_only: "仅图文组",
};

const sourceNames = {
  AdvBench: "AdvBench（有害文本）",
  Alpaca: "Alpaca（无害文本）",
  "DAN-Prompts": "DAN-Prompts（越狱模板）",
  HADES: "HADES（有害图文）",
  "MM-SafetyBench": "MM-SafetyBench（有害图文）",
  "MM-Vet": "MM-Vet（无害图文）",
  "VizWiz-VQA": "VizWiz-VQA（无害图文）",
  OpenAssistant: "OpenAssistant（无害文本）",
  XSTest: "XSTest",
  "XSTest-safe": "XSTest：无害",
  "XSTest-unsafe": "XSTest：有害",
  "CS-DJ": "CS-DJ（复杂多图干扰）",
  FigStep: "FigStep（文字排版）",
  JOOD: "JOOD（图像混合）",
};

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&apos;");
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const input = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    if (quoted) {
      if (char === '"') {
        if (input[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field !== "" || row.length) {
    row.push(field.replace(/\r$/, ""));
    if (row.some((value) => value !== "")) rows.push(row);
  }
  if (rows.length < 2) return [];
  const header = rows[0];
  return rows.slice(1).map((values) => Object.fromEntries(header.map((name, index) => [name, values[index] ?? ""])));
}

function readCsv(relativePath) {
  const filePath = path.join(resultRoot, relativePath);
  if (!fs.existsSync(filePath)) return [];
  return parseCsv(fs.readFileSync(filePath, "utf8"));
}

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(path.join(resultRoot, relativePath), "utf8"));
}

function numeric(value) {
  const result = Number(value);
  return Number.isFinite(result) ? result : Number.NaN;
}

function truthy(value) {
  return ["true", "1", "yes"].includes(String(value).toLowerCase());
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "";
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(digits);
}

function interpolateColor(left, right, amount) {
  const parse = (hex) => [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
  const a = parse(left);
  const b = parse(right);
  const values = a.map((value, index) => Math.round(value + (b[index] - value) * amount));
  return `#${values.map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

function heatColor(value, minimum, maximum) {
  if (!Number.isFinite(value)) return "#E5E7EB";
  const span = Math.max(maximum - minimum, 1e-12);
  const position = Math.max(0, Math.min(1, (value - minimum) / span));
  if (position <= 0.5) return interpolateColor("#365FA8", "#F2F1ED", position * 2);
  return interpolateColor("#F2F1ED", "#C83349", (position - 0.5) * 2);
}

function svgDocument(width, height, title, description, body) {
  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="chart-title chart-desc">
  <title id="chart-title">${escapeXml(title)}</title>
  <desc id="chart-desc">${escapeXml(description)}</desc>
  <rect width="100%" height="100%" fill="#FFFFFF"/>
  <g font-family="${fontFamily}" fill="#202124">
    ${body}
  </g>
</svg>`;
}

async function savePng(svg, relativePath) {
  const outputPath = path.join(resultRoot, relativePath);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  await sharp(Buffer.from(svg)).png({ compressionLevel: 9 }).toFile(outputPath);
}

function polyline(points, color, width = 3) {
  if (points.length < 2) return "";
  return `<polyline points="${points.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ")}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round"/>`;
}

function linePanel({ box, title, xLabel, yLabel, series, yMinimum = 0, yMaximum = 1.02, legendColumns = 1, emptyMessage = "" }) {
  const { x, y, width, height } = box;
  const sx = (value) => x + Math.max(0, Math.min(1, value)) * width;
  const sy = (value) => y + height - ((value - yMinimum) / (yMaximum - yMinimum)) * height;
  const parts = [];
  parts.push(`<text x="${x + width / 2}" y="${y - 26}" text-anchor="middle" font-size="25" font-weight="500">${escapeXml(title)}</text>`);
  for (let tick = 0; tick <= 5; tick += 1) {
    const value = tick / 5;
    const tx = sx(value);
    const ty = sy(value);
    parts.push(`<line x1="${tx}" y1="${y}" x2="${tx}" y2="${y + height}" stroke="#D8DADD" stroke-width="1"/>`);
    parts.push(`<line x1="${x}" y1="${ty}" x2="${x + width}" y2="${ty}" stroke="#D8DADD" stroke-width="1"/>`);
    parts.push(`<text x="${tx}" y="${y + height + 28}" text-anchor="middle" font-size="17">${value.toFixed(1)}</text>`);
    parts.push(`<text x="${x - 15}" y="${ty + 6}" text-anchor="end" font-size="17">${value.toFixed(1)}</text>`);
  }
  parts.push(`<line x1="${x}" y1="${y}" x2="${x}" y2="${y + height}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<line x1="${x}" y1="${y + height}" x2="${x + width}" y2="${y + height}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<text x="${x + width / 2}" y="${y + height + 62}" text-anchor="middle" font-size="20">${escapeXml(xLabel)}</text>`);
  parts.push(`<text x="${x - 58}" y="${y + height / 2}" text-anchor="middle" font-size="20" transform="rotate(-90 ${x - 58} ${y + height / 2})">${escapeXml(yLabel)}</text>`);

  series.forEach((entry) => {
    const points = entry.points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)).sort((a, b) => a.x - b.x);
    parts.push(polyline(points.map((point) => [sx(point.x), sy(point.y)]), entry.color, 3.2));
    points.filter((point) => point.selected).forEach((point) => {
      parts.push(`<circle cx="${sx(point.x)}" cy="${sy(point.y)}" r="7" fill="${entry.color}" stroke="#111111" stroke-width="2"/>`);
    });
  });

  if (!series.length && emptyMessage) {
    parts.push(`<text x="${x + width / 2}" y="${y + height / 2}" text-anchor="middle" font-size="23" fill="#555555">${escapeXml(emptyMessage)}</text>`);
  }

  const legendY = y + height + 90;
  const columnWidth = width / Math.max(legendColumns, 1);
  series.forEach((entry, index) => {
    const column = index % legendColumns;
    const row = Math.floor(index / legendColumns);
    const lx = x + column * columnWidth;
    const ly = legendY + row * 29;
    parts.push(`<line x1="${lx}" y1="${ly}" x2="${lx + 34}" y2="${ly}" stroke="${entry.color}" stroke-width="4"/>`);
    parts.push(`<text x="${lx + 44}" y="${ly + 6}" font-size="17">${escapeXml(entry.label)}</text>`);
  });
  return parts.join("\n");
}

async function renderLayerCurves({ probeRows, attackRows, outputPath, figureTitle, leftTitle }) {
  const width = 1680;
  const height = 760;
  const readouts = [...new Set(probeRows.map((row) => row.readout))].sort((a, b) => Object.keys(readoutNames).indexOf(a) - Object.keys(readoutNames).indexOf(b));
  const leftSeries = readouts.map((readout, index) => ({
    label: readoutNames[readout] || readout,
    color: palette[index % palette.length],
    points: probeRows.filter((row) => row.readout === readout).map((row) => ({
      x: numeric(row.normalized_depth),
      y: numeric(row.test_auroc),
      selected: truthy(row.validation_selected),
    })),
  }));
  const depthByLayer = new Map(probeRows.filter((row) => row.readout === "last").map((row) => [String(row.layer), numeric(row.normalized_depth)]));
  const filteredAttacks = attackRows.filter((row) => row.readout === "last");
  const attacks = [...new Set(filteredAttacks.map((row) => row.attack))].sort();
  const rightSeries = attacks.map((attack, index) => ({
    label: attackNames[attack] || attack,
    color: palette[index % palette.length],
    points: filteredAttacks.filter((row) => row.attack === attack).map((row) => ({
      x: depthByLayer.get(String(row.layer)),
      y: numeric(row.tpr),
      selected: false,
    })),
  }));
  const body = `
    <text x="${width / 2}" y="42" text-anchor="middle" font-size="30" font-weight="500">${escapeXml(figureTitle)}</text>
    ${linePanel({ box: { x: 110, y: 110, width: 620, height: 440 }, title: leftTitle, xLabel: "相对层深度", yLabel: "排序分数", series: leftSeries, legendColumns: 1 })}
    ${linePanel({ box: { x: 940, y: 110, width: 620, height: 440 }, title: "固定门槛下的视觉攻击检出率", xLabel: "相对层深度", yLabel: "攻击检出率", series: rightSeries, legendColumns: 1 })}
  `;
  await savePng(svgDocument(width, height, figureTitle, "左侧展示常规测试排序，右侧展示固定门槛下三种视觉攻击的检出率。", body), outputPath);
}

function heatmapBody({ rows, rowField, columnField, valueField, title, colorbarLabel, minimum, maximum }) {
  const rowKeys = [...new Set(rows.map((row) => row[rowField]))];
  const columnKeys = [...new Set(rows.map((row) => Number(row[columnField])))].sort((a, b) => a - b);
  const rowLabels = new Map(rowKeys.map((key) => [key, attackNames[key] || sourceNames[key] || key]));
  const longestLabel = Math.max(0, ...[...rowLabels.values()].map((label) => String(label).length));
  const left = Math.max(180, Math.min(330, 45 + longestLabel * 20));
  const right = 180;
  const width = Math.max(1120, columnKeys.length * 25 + left + right);
  const height = Math.max(430, rowKeys.length * 90 + 210);
  const top = 90;
  const plotWidth = width - left - right;
  const plotHeight = height - 190;
  const cellWidth = plotWidth / columnKeys.length;
  const cellHeight = plotHeight / rowKeys.length;
  const values = new Map(rows.map((row) => [`${row[rowField]}\u0000${Number(row[columnField])}`, numeric(row[valueField])]));
  const parts = [`<text x="${width / 2}" y="42" text-anchor="middle" font-size="30" font-weight="500">${escapeXml(title)}</text>`];
  rowKeys.forEach((rowKey, rowIndex) => {
    const rowY = top + rowIndex * cellHeight;
    parts.push(`<text x="${left - 16}" y="${rowY + cellHeight / 2 + 7}" text-anchor="end" font-size="19">${escapeXml(rowLabels.get(rowKey))}</text>`);
    columnKeys.forEach((columnKey, columnIndex) => {
      const value = values.get(`${rowKey}\u0000${columnKey}`);
      parts.push(`<rect x="${left + columnIndex * cellWidth}" y="${rowY}" width="${cellWidth + 0.5}" height="${cellHeight + 0.5}" fill="${heatColor(value, minimum, maximum)}"/>`);
    });
  });
  const tickStep = Math.max(1, Math.ceil(columnKeys.length / 12));
  columnKeys.forEach((columnKey, columnIndex) => {
    if (columnIndex % tickStep !== 0) return;
    const x = left + (columnIndex + 0.5) * cellWidth;
    parts.push(`<text x="${x}" y="${top + plotHeight + 28}" text-anchor="middle" font-size="16">${columnKey}</text>`);
  });
  parts.push(`<text x="${left + plotWidth / 2}" y="${top + plotHeight + 62}" text-anchor="middle" font-size="20">模型层</text>`);
  const barX = left + plotWidth + 70;
  const barHeight = plotHeight;
  const steps = 80;
  for (let index = 0; index < steps; index += 1) {
    const fraction = index / (steps - 1);
    const value = maximum - fraction * (maximum - minimum);
    parts.push(`<rect x="${barX}" y="${top + index * barHeight / steps}" width="24" height="${barHeight / steps + 1}" fill="${heatColor(value, minimum, maximum)}"/>`);
  }
  [maximum, (minimum + maximum) / 2, minimum].forEach((value, index) => {
    const y = top + index * barHeight / 2;
    parts.push(`<text x="${barX + 36}" y="${y + 6}" font-size="16">${escapeXml(formatNumber(value))}</text>`);
  });
  parts.push(`<text x="${barX + 100}" y="${top + barHeight / 2}" text-anchor="middle" font-size="18" transform="rotate(-90 ${barX + 100} ${top + barHeight / 2})">${escapeXml(colorbarLabel)}</text>`);
  return { width, height, body: parts.join("\n") };
}

async function renderHeatmap({ rows, valueField, title, colorbarLabel, outputPath, fixedRange }) {
  const finite = rows.map((row) => numeric(row[valueField])).filter(Number.isFinite);
  const minimum = fixedRange ? fixedRange[0] : Math.min(0, ...finite);
  const maximum = fixedRange ? fixedRange[1] : Math.max(0, ...finite);
  const chart = heatmapBody({ rows, rowField: "attack", columnField: "layer", valueField, title, colorbarLabel, minimum, maximum });
  await savePng(svgDocument(chart.width, chart.height, title, `${colorbarLabel}随攻击类型和模型层的变化。`, chart.body), outputPath);
}

function sourceAccuracy(row) {
  const positive = numeric(row.positive_n) > 0;
  const negative = numeric(row.negative_n) > 0;
  if (positive && !negative) return numeric(row.tpr);
  if (negative && !positive) return numeric(row.tnr);
  if (positive && negative) return numeric(row.balanced_accuracy);
  return Number.NaN;
}

async function renderSourceHeatmap({ analysisDir, outputPath, strong }) {
  const leaveOneSource = readCsv(path.join(analysisDir, "leave_one_source_out.csv"));
  const eligible = leaveOneSource.filter((row) => row.readout === "last" && truthy(row.eligible));
  let rows;
  let rowField;
  let title;
  if (eligible.length) {
    rows = eligible.map((row) => ({ ...row, class_accuracy: sourceAccuracy(row) }));
    rowField = "held_out_source";
    title = "整库留一：未见数据来源的识别表现";
  } else {
    rows = readCsv(path.join(analysisDir, "source_metrics.csv"))
      .filter((row) => row.readout === "last")
      .map((row) => ({ ...row, class_accuracy: sourceAccuracy(row) }));
    rowField = "source";
    title = strong ? "单一文本来源内部测试（不能表示跨来源）" : "各数据来源的常规测试表现";
  }
  const chart = heatmapBody({ rows, rowField, columnField: "layer", valueField: "class_accuracy", title, colorbarLabel: "正确识别比例", minimum: 0, maximum: 1 });
  await savePng(svgDocument(chart.width, chart.height, title, "每行代表一个数据来源，每列代表一个模型层。", chart.body), outputPath);
}

function extent(values) {
  const finite = values.filter(Number.isFinite);
  const minimum = Math.min(...finite);
  const maximum = Math.max(...finite);
  const padding = Math.max((maximum - minimum) * 0.08, 1);
  return [minimum - padding, maximum + padding];
}

function scatterPanel({ box, title, points, xDomain, yDomain, mode }) {
  const { x, y, width, height } = box;
  const sx = (value) => x + (value - xDomain[0]) / (xDomain[1] - xDomain[0]) * width;
  const sy = (value) => y + height - (value - yDomain[0]) / (yDomain[1] - yDomain[0]) * height;
  const parts = [`<text x="${x + width / 2}" y="${y - 28}" text-anchor="middle" font-size="24" font-weight="500">${escapeXml(title)}</text>`];
  for (let tick = 0; tick <= 4; tick += 1) {
    const tx = x + tick * width / 4;
    const ty = y + tick * height / 4;
    const xv = xDomain[0] + tick * (xDomain[1] - xDomain[0]) / 4;
    const yv = yDomain[1] - tick * (yDomain[1] - yDomain[0]) / 4;
    parts.push(`<line x1="${tx}" y1="${y}" x2="${tx}" y2="${y + height}" stroke="#E0E2E5" stroke-width="1"/>`);
    parts.push(`<line x1="${x}" y1="${ty}" x2="${x + width}" y2="${ty}" stroke="#E0E2E5" stroke-width="1"/>`);
    parts.push(`<text x="${tx}" y="${y + height + 25}" text-anchor="middle" font-size="15">${formatNumber(xv, 1)}</text>`);
    parts.push(`<text x="${x - 12}" y="${ty + 5}" text-anchor="end" font-size="15">${formatNumber(yv, 1)}</text>`);
  }
  parts.push(`<line x1="${x}" y1="${y}" x2="${x}" y2="${y + height}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<line x1="${x}" y1="${y + height}" x2="${x + width}" y2="${y + height}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<text x="${x + width / 2}" y="${y + height + 58}" text-anchor="middle" font-size="18">投影横轴（没有直接语义）</text>`);
  parts.push(`<text x="${x - 62}" y="${y + height / 2}" text-anchor="middle" font-size="18" transform="rotate(-90 ${x - 62} ${y + height / 2})">投影纵轴（没有直接语义）</text>`);

  const categories = mode === "label" ? ["0", "1"] : [...new Set(points.map((point) => point.source))].sort();
  const colors = new Map(categories.map((category, index) => [category, palette[index % palette.length]]));
  points.forEach((point) => {
    const category = mode === "label" ? String(point.label) : point.source;
    const color = colors.get(category);
    const px = sx(point.x);
    const py = sy(point.y);
    if (mode === "source" && point.attack) {
      parts.push(`<path d="M ${px - 4} ${py - 4} L ${px + 4} ${py + 4} M ${px + 4} ${py - 4} L ${px - 4} ${py + 4}" fill="none" stroke="${color}" stroke-width="2" opacity="0.68"/>`);
    } else {
      parts.push(`<circle cx="${px}" cy="${py}" r="4" fill="${color}" opacity="0.58"/>`);
    }
  });
  const legendColumns = mode === "label" ? 2 : 3;
  const legendY = y + height + 88;
  const columnWidth = width / legendColumns;
  categories.forEach((category, index) => {
    const lx = x + (index % legendColumns) * columnWidth;
    const ly = legendY + Math.floor(index / legendColumns) * 25;
    const color = colors.get(category);
    const label = mode === "label" ? (category === "1" ? "有害" : "无害") : (sourceNames[category] || category);
    parts.push(`<circle cx="${lx + 5}" cy="${ly}" r="5" fill="${color}"/>`);
    parts.push(`<text x="${lx + 18}" y="${ly + 6}" font-size="15">${escapeXml(label)}</text>`);
  });
  return parts.join("\n");
}

async function renderScatter({ pointsPath, analysisPath, outputPath, figureTitle }) {
  const rows = readCsv(pointsPath);
  const analysis = readJson(analysisPath);
  const layer = analysis.selected_layers.last.layer;
  const points = rows.map((row) => ({
    x: numeric(row.pca_1),
    y: numeric(row.pca_2),
    label: row.label,
    source: row.source,
    attack: truthy(row.is_attack),
  })).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const xDomain = extent(points.map((point) => point.x));
  const yDomain = extent(points.map((point) => point.y));
  const width = 1740;
  const height = 820;
  const body = `
    <text x="${width / 2}" y="42" text-anchor="middle" font-size="30" font-weight="500">${escapeXml(`${figureTitle}：第${layer}层二维压缩视图`)}</text>
    ${scatterPanel({ box: { x: 110, y: 110, width: 650, height: 480 }, title: "按有害与无害标签着色", points, xDomain, yDomain, mode: "label" })}
    ${scatterPanel({ box: { x: 970, y: 110, width: 650, height: 480 }, title: "按数据来源着色（叉号表示视觉攻击）", points, xDomain, yDomain, mode: "source" })}
  `;
  await savePng(svgDocument(width, height, `${figureTitle}二维压缩视图`, "坐标轴没有直接语义，点越接近只表示压缩后的表示越相似。", body), outputPath);
}

async function renderComparison({ comparisonDir, outputPath, figureTitle, leftTitle }) {
  if (!fs.existsSync(path.join(resultRoot, comparisonDir, "cross_model_layer_metrics.csv"))) return;
  const probeRows = readCsv(path.join(comparisonDir, "cross_model_layer_metrics.csv"));
  const attackRows = readCsv(path.join(comparisonDir, "cross_model_attack_metrics.csv"));
  const models = [...new Set(probeRows.map((row) => row.model))].sort();
  const leftSeries = models.map((model, index) => ({
    label: modelNames[model] || model,
    color: palette[index],
    points: probeRows.filter((row) => row.model === model).map((row) => ({
      x: numeric(row.normalized_depth),
      y: numeric(row.test_auroc),
      selected: truthy(row.validation_selected),
    })),
  }));
  const combinations = [...new Set(attackRows.map((row) => `${row.model}\u0000${row.attack}`))].sort();
  const rightSeries = combinations.map((combination, index) => {
    const [model, attack] = combination.split("\u0000");
    return {
      label: `${modelNames[model] || model}：${attackNames[attack] || attack}`,
      color: ["#2474A6", "#C63C3C", "#2D8A4E", "#7856A6", "#D9822B", "#159A9C"][index],
      points: attackRows.filter((row) => row.model === model && row.attack === attack).map((row) => ({
        x: numeric(row.normalized_depth),
        y: numeric(row.tpr),
        selected: false,
      })),
    };
  });
  const width = 1740;
  const height = 800;
  const body = `
    <text x="${width / 2}" y="42" text-anchor="middle" font-size="30" font-weight="500">${escapeXml(figureTitle)}</text>
    ${linePanel({ box: { x: 110, y: 110, width: 640, height: 430 }, title: leftTitle, xLabel: "相对层深度", yLabel: "排序分数", series: leftSeries, legendColumns: 1 })}
    ${linePanel({ box: { x: 980, y: 110, width: 640, height: 430 }, title: "两个模型的固定门槛攻击检出率", xLabel: "相对层深度", yLabel: "攻击检出率", series: rightSeries, legendColumns: 2, emptyMessage: "该分组没有视觉攻击样本" })}
  `;
  await savePng(svgDocument(width, height, figureTitle, "比较通义千问模型和杰玛模型的常规排序与视觉攻击检出率。", body), outputPath);
}

async function renderPanelComposition({ analysisPath, outputPath, panel }) {
  const analysis = readJson(analysisPath);
  const composition = analysis.coverage.standard_by_source_label || {};
  const rows = Object.entries(composition).map(([source, counts]) => ({
    source,
    benign: numeric(counts["0"] || 0),
    harmful: numeric(counts["1"] || 0),
  }));
  const maximum = Math.max(1, ...rows.flatMap((row) => [row.benign, row.harmful]));
  const width = 1500;
  const left = 310;
  const top = 105;
  const rowHeight = 72;
  const plotWidth = 1050;
  const plotHeight = rows.length * rowHeight;
  const height = top + plotHeight + 115;
  const sx = (value) => left + value / maximum * plotWidth;
  const parts = [
    `<text x="${width / 2}" y="42" text-anchor="middle" font-size="30" font-weight="500">${escapeXml(`${panelNames[panel]}：常规数据的来源与标签构成`)}</text>`,
  ];
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = maximum * tick / 4;
    const x = sx(value);
    parts.push(`<line x1="${x}" y1="${top}" x2="${x}" y2="${top + plotHeight}" stroke="#D8DADD" stroke-width="1"/>`);
    parts.push(`<text x="${x}" y="${top + plotHeight + 30}" text-anchor="middle" font-size="17">${Math.round(value)}</text>`);
  }
  rows.forEach((row, index) => {
    const center = top + index * rowHeight + rowHeight / 2;
    parts.push(`<text x="${left - 18}" y="${center + 7}" text-anchor="end" font-size="19">${escapeXml(sourceNames[row.source] || row.source)}</text>`);
    parts.push(`<rect x="${left}" y="${center - 23}" width="${Math.max(0, sx(row.benign) - left)}" height="20" fill="#2474A6"/>`);
    parts.push(`<rect x="${left}" y="${center + 3}" width="${Math.max(0, sx(row.harmful) - left)}" height="20" fill="#C63C3C"/>`);
  });
  parts.push(`<line x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<line x1="${left}" y1="${top + plotHeight}" x2="${left + plotWidth}" y2="${top + plotHeight}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<text x="${left + plotWidth / 2}" y="${top + plotHeight + 70}" text-anchor="middle" font-size="19">训练、验证和常规测试中的样本数（不含外部攻击）</text>`);
  parts.push(`<rect x="${width - 330}" y="62" width="28" height="15" fill="#2474A6"/><text x="${width - 292}" y="76" font-size="17">无害</text>`);
  parts.push(`<rect x="${width - 220}" y="62" width="28" height="15" fill="#C63C3C"/><text x="${width - 182}" y="76" font-size="17">有害</text>`);
  await savePng(
    svgDocument(width, height, `${panelNames[panel]}数据构成`, "横向条形图展示每个常规数据来源分别提供多少无害和有害样本。外部攻击不计入。", parts.join("\n")),
    outputPath,
  );
}

async function renderPanelSourceHeatmap({ analysisDir, outputPath, modelLabel, panel }) {
  const rows = readCsv(path.join(analysisDir, "leave_one_source_out_label_metrics.csv"))
    .filter((row) => row.readout === "last" && truthy(row.eligible));
  const analysis = readJson(path.join(analysisDir, "analysis.json"));
  const selectedLayer = analysis.selected_layers.last.layer;
  const title = `${modelLabel}：${panelNames[panel]}整库留一（选中第${selectedLayer}层）`;
  const chart = heatmapBody({
    rows,
    rowField: "held_out_source_label",
    columnField: "layer",
    valueField: "label_recall",
    title,
    colorbarLabel: "固定门槛正确识别比例",
    minimum: 0,
    maximum: 1,
  });
  await savePng(
    svgDocument(chart.width, chart.height, title, "每一行是一个完整留到测试阶段的数据来源与标签组合，每一列是一个模型层。", chart.body),
    outputPath,
  );
}

async function renderPanelAttackDomain({ analysisDir, outputPath, modelLabel, panel }) {
  const rows = readCsv(path.join(analysisDir, "attack_shift_metrics.csv"))
    .filter((row) => row.readout === "last");
  await renderHeatmap({
    rows,
    valueField: "group_cv_harmful_domain_auroc",
    title: `${modelLabel}：${panelNames[panel]}中视觉攻击与常规有害输入的可区分程度`,
    colorbarLabel: "区分分数",
    outputPath,
    fixedRange: [0, 1],
  });
}

async function renderMmSafetyVariantLoso({ inputPath, outputPath }) {
  if (!fs.existsSync(path.join(resultRoot, inputPath))) return;
  const rows = readCsv(inputPath).filter((row) =>
    row.held_out_source === "MM-SafetyBench"
    && row.panel === "multimodal_only"
    && row.grouping === "variant"
  );
  if (!rows.length) return;
  const variants = ["SD", "TYPO", "SD_TYPO"].filter((variant) => rows.some((row) => row.subgroup === variant));
  const variantLabels = {
    SD: "生成图像",
    TYPO: "文字嵌入图像",
    SD_TYPO: "生成图像与文字组合",
  };
  const models = ["qwen25vl7b", "gemma3_12b"].filter((model) => rows.some((row) => row.model === model));
  const values = new Map(rows.map((row) => [`${row.model}\u0000${row.subgroup}`, row]));
  const width = 1380;
  const height = 760;
  const left = 120;
  const right = 70;
  const top = 130;
  const bottom = 150;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const groupWidth = plotWidth / variants.length;
  const barWidth = Math.min(120, groupWidth / Math.max(models.length + 1, 3));
  const sy = (value) => top + plotHeight - Math.max(0, Math.min(1, value)) * plotHeight;
  const parts = [
    `<text x="${width / 2}" y="42" text-anchor="middle" font-size="30" font-weight="500">MM-SafetyBench 仅图文整库留一：不同输入类型</text>`,
    `<text x="${width / 2}" y="78" text-anchor="middle" font-size="18" fill="#555555">训练时完全移除 MM-SafetyBench，再按其原始类型统计有害检出率</text>`,
  ];
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = tick / 4;
    const y = sy(value);
    parts.push(`<line x1="${left}" y1="${y}" x2="${left + plotWidth}" y2="${y}" stroke="#D8DADD" stroke-width="1"/>`);
    parts.push(`<text x="${left - 16}" y="${y + 6}" text-anchor="end" font-size="17">${Math.round(value * 100)}%</text>`);
  }
  variants.forEach((variant, variantIndex) => {
    const center = left + groupWidth * (variantIndex + 0.5);
    const totalBarsWidth = models.length * barWidth;
    models.forEach((model, modelIndex) => {
      const row = values.get(`${model}\u0000${variant}`);
      if (!row) return;
      const value = numeric(row.label_recall);
      const n = numeric(row.n);
      const x = center - totalBarsWidth / 2 + modelIndex * barWidth;
      const y = sy(value);
      const color = palette[modelIndex];
      parts.push(`<rect x="${x + 10}" y="${y}" width="${barWidth - 20}" height="${top + plotHeight - y}" fill="${color}"/>`);
      parts.push(`<text x="${x + barWidth / 2}" y="${Math.max(top + 18, y - 10)}" text-anchor="middle" font-size="18" font-weight="500">${(value * 100).toFixed(1)}%</text>`);
      parts.push(`<text x="${x + barWidth / 2}" y="${top + plotHeight + 25}" text-anchor="middle" font-size="15" fill="#555555">n=${Math.round(n)}</text>`);
    });
    parts.push(`<text x="${center}" y="${top + plotHeight + 62}" text-anchor="middle" font-size="21">${escapeXml(variantLabels[variant] || variant)}</text>`);
  });
  parts.push(`<line x1="${left}" y1="${top}" x2="${left}" y2="${top + plotHeight}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<line x1="${left}" y1="${top + plotHeight}" x2="${left + plotWidth}" y2="${top + plotHeight}" stroke="#333333" stroke-width="2"/>`);
  parts.push(`<text x="42" y="${top + plotHeight / 2}" text-anchor="middle" font-size="20" transform="rotate(-90 42 ${top + plotHeight / 2})">有害检出率</text>`);
  models.forEach((model, index) => {
    const x = width / 2 - 260 + index * 300;
    parts.push(`<rect x="${x}" y="${height - 46}" width="28" height="15" fill="${palette[index]}"/>`);
    parts.push(`<text x="${x + 40}" y="${height - 32}" font-size="17">${escapeXml(modelNames[model] || model)}</text>`);
  });
  await savePng(
    svgDocument(
      width,
      height,
      "MM-SafetyBench 仅图文整库留一分类型结果",
      "每组柱形比较两个模型在完全留出 MM-SafetyBench 后，对生成图像、文字嵌入图像、生成图像与文字组合三类图文输入的有害检出率。",
      parts.join("\n"),
    ),
    outputPath,
  );
}

async function renderModelFigures(config) {
  if (!fs.existsSync(path.join(resultRoot, config.analysisDir, "analysis.json"))) return;
  const probeRows = readCsv(path.join(config.analysisDir, "layer_probe_metrics.csv"));
  const attackRows = readCsv(path.join(config.analysisDir, "attack_shift_metrics.csv"));
  await renderLayerCurves({
    probeRows,
    attackRows,
    outputPath: path.join(config.outputDir, "layer_curves.png"),
    figureTitle: `${config.modelLabel}：${config.modeLabel}`,
    leftTitle: config.strong ? "显式安全文本内部的有害与无害排序" : "常规测试中的有害与无害排序",
  });

  const lastAttackRows = attackRows.filter((row) => row.readout === "last");
  await renderHeatmap({
    rows: lastAttackRows,
    valueField: "standardized_score_shift",
    title: `${config.modelLabel}：视觉攻击的分数位置偏移`,
    colorbarLabel: "相对常规有害分数的偏移（以标准差计）",
    outputPath: path.join(config.outputDir, "attack_score_shift_heatmap.png"),
  });
  await renderHeatmap({
    rows: lastAttackRows,
    valueField: "group_cv_harmful_domain_auroc",
    title: `${config.modelLabel}：视觉攻击与常规有害输入的可区分程度`,
    colorbarLabel: "区分分数",
    outputPath: path.join(config.outputDir, "attack_domain_auroc_heatmap.png"),
    fixedRange: [0, 1],
  });
  await renderSourceHeatmap({
    analysisDir: config.analysisDir,
    outputPath: path.join(config.outputDir, "source_generalization_heatmap.png"),
    strong: config.strong,
  });
  await renderScatter({
    pointsPath: config.pointsPath,
    analysisPath: path.join(config.analysisDir, "analysis.json"),
    outputPath: path.join(config.outputDir, "selected_layer_pca.png"),
    figureTitle: `${config.modelLabel}：${config.modeLabel}`,
  });

  const commonProbe = readCsv(path.join(config.analysisDir, "common_panel_layer_metrics.csv"));
  if (commonProbe.length) {
    await renderLayerCurves({
      probeRows: commonProbe,
      attackRows: readCsv(path.join(config.analysisDir, "common_panel_attack_metrics.csv")),
      outputPath: path.join(config.outputDir, "common_multimodal_panel_layer_curves.png"),
      figureTitle: `${config.modelLabel}：相同多模态样本对照`,
      leftTitle: "相同多模态样本中的有害与无害排序",
    });
  }
}

async function main() {
  const configs = [
    {
      analysisDir: "analysis/qwen25vl7b",
      pointsPath: "figures/qwen25vl7b/pca_points.csv",
      outputDir: "figures_zh/qwen25vl7b",
      modelLabel: "通义千问模型",
      modeLabel: "普通结果",
      strong: false,
    },
    {
      analysisDir: "analysis/gemma3_12b",
      pointsPath: "figures/gemma3_12b/pca_points.csv",
      outputDir: "figures_zh/gemma3_12b",
      modelLabel: "杰玛模型",
      modeLabel: "普通结果",
      strong: false,
    },
    {
      analysisDir: "analysis_strong/qwen25vl7b",
      pointsPath: "figures_strong/qwen25vl7b/pca_points.csv",
      outputDir: "figures_strong_zh/qwen25vl7b",
      modelLabel: "通义千问模型",
      modeLabel: "高置信标签结果",
      strong: true,
    },
    {
      analysisDir: "analysis_strong/gemma3_12b",
      pointsPath: "figures_strong/gemma3_12b/pca_points.csv",
      outputDir: "figures_strong_zh/gemma3_12b",
      modelLabel: "杰玛模型",
      modeLabel: "高置信标签结果",
      strong: true,
    },
  ];
  for (const config of configs) await renderModelFigures(config);
  await renderComparison({
    comparisonDir: "comparison",
    outputPath: "comparison_zh/cross_model_layer_curves.png",
    figureTitle: "两个模型的普通结果比较",
    leftTitle: "两个模型的常规测试排序",
  });
  await renderComparison({
    comparisonDir: "comparison_strong",
    outputPath: "comparison_strong_zh/cross_model_layer_curves.png",
    figureTitle: "两个模型的高置信标签结果比较",
    leftTitle: "两个模型的显式安全文本内部排序",
  });
  await renderComparison({
    comparisonDir: "comparison_panels/all",
    outputPath: "comparison_panels_zh/all/cross_model_layer_curves.png",
    figureTitle: "两个模型的全部样本组逐层比较",
    leftTitle: "全部样本组常规测试排序",
  });

  for (const panel of ["all", "text_only", "multimodal_only"]) {
    await renderPanelComposition({
      analysisPath: `analysis_panels/${panel}/qwen25vl7b/analysis.json`,
      outputPath: `figures_panels_zh/${panel}/panel_composition.png`,
      panel,
    });
  }
  await renderComparison({
    comparisonDir: "comparison_panels/text_only",
    outputPath: "comparison_panels_zh/text_only/cross_model_layer_curves.png",
    figureTitle: "两个模型的纯文本组逐层比较",
    leftTitle: "纯文本组常规测试排序",
  });
  await renderComparison({
    comparisonDir: "comparison_panels/multimodal_only",
    outputPath: "comparison_panels_zh/multimodal_only/cross_model_layer_curves.png",
    figureTitle: "两个模型的仅图文组逐层比较",
    leftTitle: "仅图文组常规测试排序",
  });
  for (const panel of ["all", "text_only", "multimodal_only"]) {
    for (const model of ["qwen25vl7b", "gemma3_12b"]) {
      await renderPanelSourceHeatmap({
        analysisDir: `analysis_panels/${panel}/${model}`,
        outputPath: `figures_panels_zh/${panel}/${model}/source_generalization_heatmap.png`,
        modelLabel: `${modelNames[model]}模型`,
        panel,
      });
    }
  }
  for (const panel of ["all", "multimodal_only"]) {
    for (const model of ["qwen25vl7b", "gemma3_12b"]) {
      await renderPanelAttackDomain({
        analysisDir: `analysis_panels/${panel}/${model}`,
        outputPath: `figures_panels_zh/${panel}/${model}/attack_domain_auroc_heatmap.png`,
        modelLabel: modelNames[model],
        panel,
      });
    }
  }
  await renderMmSafetyVariantLoso({
    inputPath: "diagnostics/mm_safetybench_loso_subgroups.csv",
    outputPath: "figures_panels_zh/multimodal_only/mm_safetybench_variant_loso.png",
  });
  console.log("已生成当前结果目录中可用的中文结果图。");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
