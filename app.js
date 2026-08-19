/**
 * Attention Efficiency Explorer — Application Script
 * Live 2D Mask Generator, Plotly Interactive Charts, and Data Exports
 */

// ---------------------------------------------------------------------------
// Dataset Definitions
// ---------------------------------------------------------------------------

const DATASETS = {
  four_way_sweep: {
    "standard": [
      { "seq_len": 128, "time_mean": 0.00062146, "mem_mean": 24.2817 },
      { "seq_len": 256, "time_mean": 0.00053716, "mem_mean": 31.4692 },
      { "seq_len": 512, "time_mean": 0.00165629, "mem_mean": 55.5942 },
      { "seq_len": 1024, "time_mean": 0.00276771, "mem_mean": 153.3442 },
      { "seq_len": 2048, "time_mean": 0.00968046, "mem_mean": 540.8442 },
      { "seq_len": 4096, "time_mean": 0.03728642, "mem_mean": 2084.5942 },
      { "seq_len": 8192, "time_mean": 0.14704161, "mem_mean": 8242.5942 }
    ],
    "naive_sliding_window": [
      { "seq_len": 128, "time_mean": 0.00233698, "mem_mean": 24.4233 },
      { "seq_len": 256, "time_mean": 0.00070910, "mem_mean": 32.0337 },
      { "seq_len": 512, "time_mean": 0.00411143, "mem_mean": 58.5981 },
      { "seq_len": 1024, "time_mean": 0.00553675, "mem_mean": 162.3521 },
      { "seq_len": 2048, "time_mean": 0.01380777, "mem_mean": 577.6099 },
      { "seq_len": 4096, "time_mean": 0.05474606, "mem_mean": 2228.6255 },
      { "seq_len": 8192, "time_mean": 0.21641483, "mem_mean": 8818.6567 }
    ],
    "disjoint_block": [
      { "seq_len": 128, "time_mean": 0.00046363, "mem_mean": 22.3442 },
      { "seq_len": 256, "time_mean": 0.00042748, "mem_mean": 22.8442 },
      { "seq_len": 512, "time_mean": 0.00041389, "mem_mean": 23.8442 },
      { "seq_len": 1024, "time_mean": 0.00044866, "mem_mean": 25.8442 },
      { "seq_len": 2048, "time_mean": 0.00041561, "mem_mean": 29.8442 },
      { "seq_len": 4096, "time_mean": 0.00071936, "mem_mean": 38.5942 },
      { "seq_len": 8192, "time_mean": 0.00128727, "mem_mean": 54.5942 }
    ],
    "overlapping_sliding_window": [
      { "seq_len": 128, "time_mean": 0.00086908, "mem_mean": 22.9800 },
      { "seq_len": 256, "time_mean": 0.00073209, "mem_mean": 24.1074 },
      { "seq_len": 512, "time_mean": 0.00070534, "mem_mean": 26.3623 },
      { "seq_len": 1024, "time_mean": 0.00071654, "mem_mean": 30.8721 },
      { "seq_len": 2048, "time_mean": 0.00072832, "mem_mean": 39.8916 },
      { "seq_len": 4096, "time_mean": 0.00133557, "mem_mean": 61.1479 },
      { "seq_len": 8192, "time_mean": 0.00216050, "mem_mean": 94.9588 }
    ]
  },

  copy_task: {
    "seq_len": 128,
    "standard_accuracy": 1.0,
    "overlapping_accuracy": {
      "4": 0.017333984375,
      "8": 0.018798828125,
      "16": 0.02587890625,
      "32": 0.023193359375,
      "64": 0.01904296875,
      "128": 0.9921875
    },
    "disjoint_accuracy": {
      "4": 0.022705078125,
      "8": 0.0205078125,
      "16": 0.02294921875,
      "32": 0.017333984375,
      "64": 0.017822265625,
      "128": 0.020751953125
    }
  },

  classification: {
    "standard_accuracy": 0.98499995470047,
    "sliding_window_w32_accuracy": 0.98499995470047,
    "sliding_window_w64_accuracy": 0.9649999737739563
  },

  depth: {
    "seq_len": 128,
    "overlapping_w32_by_depth": {
      "1": 0.019287109375,
      "2": 0.0205078125,
      "3": 0.025146484375,
      "4": 0.025146484375
    },
    "standard_by_depth": {
      "1": 1.0,
      "2": 0.974609375,
      "4": 0.97802734375
    }
  }
};

// Colors matching original dashboard tokens
const COLORS = {
  ink: "#14162B",
  muted: "#6B7089",
  hairline: "#E4E5EC",
  accent: "#4338CA",
  naive: "#B45309",
  disjoint: "#0F766E"
};

const VARIANT_NAMES = {
  "standard": "Standard (full) attention",
  "naive_sliding_window": "Naive sliding-window (masked)",
  "disjoint_block": "Disjoint block",
  "overlapping_sliding_window": "Overlapping sliding-window"
};

const PLOT_FONT = { family: "IBM Plex Sans, sans-serif", size: 12, color: COLORS.ink };

// Helper to style Plotly figures consistently
function stylePlotlyFig(layout, height = 400, titleText = "") {
  return Object.assign({
    font: PLOT_FONT,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#FFFFFF",
    height: height,
    margin: { l: 45, r: 25, t: titleText ? 45 : 20, b: 50 },
    legend: {
      orientation: "h", yanchor: "top", y: -0.18, xanchor: "left", x: 0,
      font: { size: 11.5, family: "IBM Plex Sans, sans-serif" }
    },
    xaxis: {
      gridcolor: COLORS.hairline, zeroline: false, linecolor: COLORS.hairline,
      tickfont: { family: "IBM Plex Mono, monospace", size: 11 }
    },
    yaxis: {
      gridcolor: COLORS.hairline, zeroline: false, linecolor: COLORS.hairline,
      tickfont: { family: "IBM Plex Mono, monospace", size: 11 }
    }
  }, layout);
}

// ---------------------------------------------------------------------------
// 2D Attention Mask Generator
// ---------------------------------------------------------------------------

function generateMaskMatrix(variant, seqLen, windowSize) {
  const matrix = Array.from({ length: seqLen }, () => new Uint8Array(seqLen));
  
  if (variant === "Standard (full) attention") {
    for (let r = 0; r < seqLen; r++) {
      matrix[r].fill(1);
    }
    return matrix;
  }

  const numBlocks = Math.ceil(seqLen / windowSize);
  for (let b = 0; b < numBlocks; b++) {
    const q0 = b * windowSize;
    const q1 = Math.min((b + 1) * windowSize, seqLen);

    let blocks = [b];
    if (variant === "Overlapping sliding-window") {
      blocks = [b - 1, b, b + 1].filter(x => x >= 0 && x < numBlocks);
    }

    for (const ab of blocks) {
      const k0 = ab * windowSize;
      const k1 = Math.min((ab + 1) * windowSize, seqLen);
      for (let q = q0; q < q1; q++) {
        for (let k = k0; k < k1; k++) {
          matrix[q][k] = 1;
        }
      }
    }
  }

  return matrix;
}

// ---------------------------------------------------------------------------
// Render Overview Preview
// ---------------------------------------------------------------------------

function renderOverviewPreview(variant) {
  const mask = generateMaskMatrix(variant, 48, 8);
  const zData = mask.map(row => Array.from(row));

  const data = [{
    z: zData,
    type: 'heatmap',
    colorscale: [[0, '#F1F1F6'], [1, COLORS.accent]],
    showscale: false
  }];

  const layout = stylePlotlyFig({
    xaxis: { title: "Key position", tickfont: { family: "IBM Plex Mono", size: 10 } },
    yaxis: { title: "Query position", tickfont: { family: "IBM Plex Mono", size: 10 }, autorange: 'reversed' }
  }, 320);

  Plotly.newPlot('overview-preview-chart', data, layout, { responsive: true, displayModeBar: false });
}

// ---------------------------------------------------------------------------
// Render Attention Patterns Tab Matrix
// ---------------------------------------------------------------------------

function updatePatternPanel() {
  const seqLen = parseInt(document.getElementById('pattern-seq-len').value, 10);
  const winSize = parseInt(document.getElementById('pattern-win-size').value, 10);
  const variant = document.getElementById('pattern-variant').value;

  document.getElementById('val-seq-len').textContent = seqLen;
  document.getElementById('val-win-size').textContent = winSize;

  const mask = generateMaskMatrix(variant, seqLen, winSize);
  const totalPairs = seqLen * seqLen;
  let countAttended = 0;
  for (let r = 0; r < seqLen; r++) {
    for (let c = 0; c < seqLen; c++) {
      if (mask[r][c] === 1) countAttended++;
    }
  }

  const coveragePct = ((countAttended / totalPairs) * 100).toFixed(1);
  const numBlocks = Math.ceil(seqLen / winSize);

  document.getElementById('pattern-coverage').textContent = `${coveragePct}%`;
  document.getElementById('pattern-coverage-note').textContent = `${countAttended.toLocaleString()} of ${totalPairs.toLocaleString()} query–key pairs`;
  document.getElementById('pattern-blocks').textContent = numBlocks;

  const reachTile = document.getElementById('pattern-reach-tile');
  if (variant === "Standard (full) attention") {
    reachTile.style.display = 'none';
  } else {
    reachTile.style.display = 'flex';
    const reach = winSize * (variant === "Overlapping sliding-window" ? 2 : 1);
    document.getElementById('pattern-reach').textContent = `${reach} tokens`;
  }

  const degenWarning = document.getElementById('degenerate-warning');
  if (variant === "Overlapping sliding-window" && numBlocks <= 3) {
    degenWarning.classList.remove('hidden');
  } else {
    degenWarning.classList.add('hidden');
  }

  // Plot Heatmap Matrix
  const zData = mask.map(row => Array.from(row));
  const data = [{
    z: zData,
    type: 'heatmap',
    colorscale: [[0, '#F1F1F6'], [1, COLORS.accent]],
    showscale: false
  }];

  const layout = stylePlotlyFig({
    xaxis: { title: "Key position" },
    yaxis: { title: "Query position", autorange: 'reversed' }
  }, 440);

  Plotly.newPlot('mask-matrix-chart', data, layout, { responsive: true, displayModeBar: true });
}

// ---------------------------------------------------------------------------
// Render Efficiency Scaling Tab
// ---------------------------------------------------------------------------

function renderScalingTab() {
  const measure = document.querySelector('input[name="scaling-measure"]:checked').value;
  const sweep = DATASETS.four_way_sweep;

  const varColors = {
    "standard": COLORS.ink,
    "naive_sliding_window": COLORS.naive,
    "disjoint_block": COLORS.disjoint,
    "overlapping_sliding_window": COLORS.accent
  };

  function createTraces(yKey) {
    return Object.keys(sweep).map(key => {
      return {
        x: sweep[key].map(d => d.seq_len),
        y: sweep[key].map(d => d[yKey]),
        mode: 'lines+markers',
        name: VARIANT_NAMES[key],
        line: { color: varColors[key], width: 2 },
        marker: { size: 6 }
      };
    });
  }

  const chartWrapper = document.getElementById('scaling-charts-wrapper');
  const runtimeBox = document.getElementById('runtime-chart-box');
  const memoryBox = document.getElementById('memory-chart-box');

  if (measure === 'Both') {
    chartWrapper.style.gridTemplateColumns = '1fr 1fr';
    runtimeBox.style.display = 'block';
    memoryBox.style.display = 'block';

    const layoutRuntime = stylePlotlyFig({
      title: { text: "Runtime", font: { family: "IBM Plex Serif", size: 15, color: COLORS.ink }, x: 0 },
      xaxis: { type: 'log', title: "Sequence length" },
      yaxis: { type: 'log', title: "Seconds" }
    }, 400, "Runtime");

    const layoutMem = stylePlotlyFig({
      title: { text: "Peak memory", font: { family: "IBM Plex Serif", size: 15, color: COLORS.ink }, x: 0 },
      xaxis: { type: 'log', title: "Sequence length" },
      yaxis: { type: 'log', title: "MB" }
    }, 400, "Peak memory");

    Plotly.newPlot('scaling-runtime-chart', createTraces('time_mean'), layoutRuntime, { responsive: true });
    Plotly.newPlot('scaling-memory-chart', createTraces('mem_mean'), layoutMem, { responsive: true });

  } else if (measure === 'Runtime') {
    chartWrapper.style.gridTemplateColumns = '1fr';
    runtimeBox.style.display = 'block';
    memoryBox.style.display = 'none';

    const layoutRuntime = stylePlotlyFig({
      title: { text: "Runtime", font: { family: "IBM Plex Serif", size: 15, color: COLORS.ink }, x: 0 },
      xaxis: { type: 'log', title: "Sequence length" },
      yaxis: { type: 'log', title: "Seconds" }
    }, 440, "Runtime");

    Plotly.newPlot('scaling-runtime-chart', createTraces('time_mean'), layoutRuntime, { responsive: true });

  } else {
    chartWrapper.style.gridTemplateColumns = '1fr';
    runtimeBox.style.display = 'none';
    memoryBox.style.display = 'block';

    const layoutMem = stylePlotlyFig({
      title: { text: "Peak memory", font: { family: "IBM Plex Serif", size: 15, color: COLORS.ink }, x: 0 },
      xaxis: { type: 'log', title: "Sequence length" },
      yaxis: { type: 'log', title: "MB" }
    }, 440, "Peak memory");

    Plotly.newPlot('scaling-memory-chart', createTraces('mem_mean'), layoutMem, { responsive: true });
  }

  // Populate 8,192 tiles
  const tilesContainer = document.getElementById('scaling-tiles-container');
  tilesContainer.innerHTML = '';

  const stdMem = sweep.standard.find(d => d.seq_len === 8192).mem_mean;

  Object.keys(sweep).forEach(key => {
    const d8192 = sweep[key].find(d => d.seq_len === 8192);
    const ratio = stdMem / d8192.mem_mean;
    
    let note = "";
    if (key === "standard") note = "reference";
    else if (ratio < 1.05) note = "no saving over standard";
    else note = `${Math.round(ratio)}× lighter than standard`;

    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.style.borderLeftColor = varColors[key];
    tile.innerHTML = `
      <div class="tile-label">${VARIANT_NAMES[key]}</div>
      <div class="tile-value">${Math.round(d8192.mem_mean)} MB</div>
      <div class="tile-note">${(d8192.time_mean * 1000).toFixed(1)} ms &middot; ${note}</div>
    `;
    tilesContainer.appendChild(tile);
  });

  // Populate Data Table
  const tableBody = document.querySelector('#sweep-data-table tbody');
  tableBody.innerHTML = '';
  Object.keys(sweep).forEach(key => {
    sweep[key].forEach(p => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${VARIANT_NAMES[key]}</td>
        <td>${p.seq_len}</td>
        <td>${p.time_mean.toFixed(6)}</td>
        <td>${p.mem_mean.toFixed(4)}</td>
      `;
      tableBody.appendChild(tr);
    });
  });
}

// ---------------------------------------------------------------------------
// Render Task Accuracy Tab
// ---------------------------------------------------------------------------

function renderTaskAccuracyTab() {
  const selectedTask = document.querySelector('input[name="accuracy-task"]:checked').value;
  const copyContainer = document.getElementById('task-copy-container');
  const classContainer = document.getElementById('task-classification-container');

  if (selectedTask === 'copy') {
    copyContainer.classList.remove('hidden');
    classContainer.classList.add('hidden');

    const copyData = DATASETS.copy_task;
    const windows = ["4", "8", "16", "32", "64", "128"];
    
    const traceOv = {
      x: windows,
      y: windows.map(w => copyData.overlapping_accuracy[w]),
      name: "Overlapping sliding-window",
      type: 'bar',
      marker: { color: COLORS.accent }
    };

    const traceDis = {
      x: windows,
      y: windows.map(w => copyData.disjoint_accuracy[w]),
      name: "Disjoint block",
      type: 'bar',
      marker: { color: COLORS.disjoint }
    };

    const layout = stylePlotlyFig({
      title: { text: "Copy accuracy by window size", font: { family: "IBM Plex Serif", size: 15 }, x: 0 },
      barmode: 'group',
      xaxis: { title: "Window size", type: 'category' },
      yaxis: { title: "Accuracy", range: [0, 1.08] },
      shapes: [
        {
          type: 'line', x0: -0.5, x1: 5.5, y0: copyData.standard_accuracy, y1: copyData.standard_accuracy,
          line: { color: COLORS.ink, width: 1.5, dash: 'dash' }
        },
        {
          type: 'line', x0: -0.5, x1: 5.5, y0: 0.02, y1: 0.02,
          line: { color: COLORS.muted, width: 1, dash: 'dot' }
        },
        {
          type: 'rect', x0: 4.5, x1: 5.5, y0: 0, y1: 1.08,
          fillcolor: COLORS.muted, opacity: 0.1, line: { width: 0 }
        }
      ],
      annotations: [
        {
          x: 0, y: copyData.standard_accuracy, text: "Standard attention · 1.00",
          showarrow: false, yanchor: "bottom", font: { family: "IBM Plex Mono", size: 11, color: COLORS.ink }
        },
        {
          x: 0, y: 0.02, text: "Chance",
          showarrow: false, yanchor: "bottom", font: { family: "IBM Plex Mono", size: 11, color: COLORS.muted }
        },
        {
          x: 5, y: 1.02, text: "degenerate",
          showarrow: false, font: { family: "IBM Plex Mono", size: 11, color: COLORS.muted }
        }
      ]
    }, 440, "Copy accuracy");

    Plotly.newPlot('copy-accuracy-chart', [traceOv, traceDis], layout, { responsive: true });

    // Table
    const tableBody = document.querySelector('#copy-data-table tbody');
    tableBody.innerHTML = '';
    windows.forEach(w => {
      tableBody.innerHTML += `
        <tr><td>Overlapping sliding-window</td><td>${w}</td><td>${copyData.overlapping_accuracy[w].toFixed(6)}</td></tr>
        <tr><td>Disjoint block</td><td>${w}</td><td>${copyData.disjoint_accuracy[w].toFixed(6)}</td></tr>
      `;
    });

  } else {
    copyContainer.classList.add('hidden');
    classContainer.classList.remove('hidden');

    const classData = DATASETS.classification;
    const configs = [
      { name: "Standard (full) attention", val: classData.standard_accuracy, color: COLORS.ink },
      { name: "Disjoint block, w=32", val: classData.sliding_window_w32_accuracy, color: COLORS.disjoint },
      { name: "Disjoint block, w=64", val: classData.sliding_window_w64_accuracy, color: COLORS.disjoint }
    ];

    const trace = {
      x: configs.map(c => c.name),
      y: configs.map(c => c.val),
      type: 'bar',
      text: configs.map(c => c.val.toFixed(3)),
      textposition: 'auto',
      marker: { color: configs.map(c => c.color) },
      textfont: { family: "IBM Plex Mono", size: 12 }
    };

    const layout = stylePlotlyFig({
      title: { text: "Classification accuracy", font: { family: "IBM Plex Serif", size: 15 }, x: 0 },
      xaxis: { title: "" },
      yaxis: { title: "Accuracy", range: [0, 1.08] }
    }, 400, "Classification accuracy");

    Plotly.newPlot('classification-accuracy-chart', [trace], layout, { responsive: true });

    // Table
    const tableBody = document.querySelector('#classification-data-table tbody');
    tableBody.innerHTML = '';
    configs.forEach(c => {
      tableBody.innerHTML += `<tr><td>${c.name}</td><td>${c.val.toFixed(6)}</td></tr>`;
    });
  }
}

// ---------------------------------------------------------------------------
// Render Depth Study Tab
// ---------------------------------------------------------------------------

function renderDepthStudyTab() {
  const depthData = DATASETS.depth;

  const traceOv = {
    x: Object.keys(depthData.overlapping_w32_by_depth).map(Number),
    y: Object.values(depthData.overlapping_w32_by_depth),
    mode: 'lines+markers',
    name: "Overlapping sliding-window",
    line: { color: COLORS.accent, width: 2 },
    marker: { size: 7 }
  };

  const traceStd = {
    x: Object.keys(depthData.standard_by_depth).map(Number),
    y: Object.values(depthData.standard_by_depth),
    mode: 'lines+markers',
    name: "Standard (full) attention",
    line: { color: COLORS.ink, width: 2 },
    marker: { size: 7 }
  };

  const layout = stylePlotlyFig({
    title: { text: "Copy accuracy against number of layers", font: { family: "IBM Plex Serif", size: 15 }, x: 0 },
    xaxis: { title: "Attention layers", dtick: 1 },
    yaxis: { title: "Accuracy", range: [0, 1.08] },
    shapes: [
      {
        type: 'line', x0: 0.5, x1: 4.5, y0: 0.02, y1: 0.02,
        line: { color: COLORS.muted, width: 1, dash: 'dot' }
      }
    ],
    annotations: [
      {
        x: 1, y: 0.02, text: "Chance",
        showarrow: false, yanchor: "bottom", font: { family: "IBM Plex Mono", size: 11, color: COLORS.muted }
      }
    ]
  }, 440, "Depth study");

  Plotly.newPlot('depth-study-chart', [traceOv, traceStd], layout, { responsive: true });

  // Table
  const tableBody = document.querySelector('#depth-data-table tbody');
  tableBody.innerHTML = '';
  Object.keys(depthData.overlapping_w32_by_depth).forEach(d => {
    tableBody.innerHTML += `<tr><td>Overlapping sliding-window (w=32)</td><td>${d}</td><td>${depthData.overlapping_w32_by_depth[d].toFixed(6)}</td></tr>`;
  });
  Object.keys(depthData.standard_by_depth).forEach(d => {
    tableBody.innerHTML += `<tr><td>Standard (full) attention</td><td>${d}</td><td>${depthData.standard_by_depth[d].toFixed(6)}</td></tr>`;
  });
}

// ---------------------------------------------------------------------------
// Data & Exports Tab Inspector
// ---------------------------------------------------------------------------

function renderExportsInspector() {
  const container = document.getElementById('files-inspector-container');
  container.innerHTML = '';

  const filesMap = [
    { name: "classification_results.json", data: DATASETS.classification },
    { name: "copy_task_v2_results.json", data: DATASETS.copy_task },
    { name: "depth_results.json", data: DATASETS.depth },
    { name: "four_way_sweep_results.json", data: DATASETS.four_way_sweep }
  ];

  filesMap.forEach(f => {
    const jsonStr = JSON.stringify(f.data, null, 2);
    const sizeBytes = new Blob([jsonStr]).size;

    const card = document.createElement('div');
    card.className = 'file-card';
    card.innerHTML = `
      <div class="file-header">
        <span class="file-title">📄 ${f.name}</span>
        <span class="file-meta">${sizeBytes.toLocaleString()} bytes</span>
      </div>
      <pre class="json-code-block">${jsonStr}</pre>
      <button class="btn btn-secondary btn-sm" onclick="downloadJSON('${f.name}')">Download ${f.name}</button>
    `;
    container.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Download Helpers
// ---------------------------------------------------------------------------

function downloadFile(filename, text, mime = "text/csv") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadJSON(filename) {
  const keyMap = {
    "classification_results.json": DATASETS.classification,
    "copy_task_v2_results.json": DATASETS.copy_task,
    "depth_results.json": DATASETS.depth,
    "four_way_sweep_results.json": DATASETS.four_way_sweep
  };
  const jsonStr = JSON.stringify(keyMap[filename], null, 2);
  downloadFile(filename, jsonStr, "application/json");
}

function downloadMaskCSV() {
  const seqLen = parseInt(document.getElementById('pattern-seq-len').value, 10);
  const winSize = parseInt(document.getElementById('pattern-win-size').value, 10);
  const variant = document.getElementById('pattern-variant').value;
  const mask = generateMaskMatrix(variant, seqLen, winSize);

  let csv = "";
  for (let r = 0; r < seqLen; r++) {
    csv += Array.from(mask[r]).join(",") + "\n";
  }
  downloadFile(`mask_n${seqLen}_w${winSize}.csv`, csv);
}

function downloadSweepCSV() {
  const sweep = DATASETS.four_way_sweep;
  let csv = "variant,seq_len,time_s,mem_MB\n";
  Object.keys(sweep).forEach(key => {
    sweep[key].forEach(p => {
      csv += `"${VARIANT_NAMES[key]}",${p.seq_len},${p.time_mean},${p.mem_mean}\n`;
    });
  });
  downloadFile("efficiency_sweep.csv", csv);
}

function downloadCopyCSV() {
  const copyData = DATASETS.copy_task;
  let csv = "variant,window,accuracy\n";
  Object.keys(copyData.overlapping_accuracy).forEach(w => {
    csv += `"Overlapping sliding-window",${w},${copyData.overlapping_accuracy[w]}\n`;
  });
  Object.keys(copyData.disjoint_accuracy).forEach(w => {
    csv += `"Disjoint block",${w},${copyData.disjoint_accuracy[w]}\n`;
  });
  downloadFile("copy_task_accuracy.csv", csv);
}

function downloadClassificationCSV() {
  const c = DATASETS.classification;
  let csv = "configuration,accuracy\n";
  csv += `"Standard (full) attention",${c.standard_accuracy}\n`;
  csv += `"Disjoint block, w=32",${c.sliding_window_w32_accuracy}\n`;
  csv += `"Disjoint block, w=64",${c.sliding_window_w64_accuracy}\n`;
  downloadFile("classification_accuracy.csv", csv);
}

function downloadDepthCSV() {
  const d = DATASETS.depth;
  let csv = "variant,depth,accuracy\n";
  Object.keys(d.overlapping_w32_by_depth).forEach(layer => {
    csv += `"Overlapping sliding-window",${layer},${d.overlapping_w32_by_depth[layer]}\n`;
  });
  Object.keys(d.standard_by_depth).forEach(layer => {
    csv += `"Standard (full) attention",${layer},${d.standard_by_depth[layer]}\n`;
  });
  downloadFile("depth_study.csv", csv);
}

// ---------------------------------------------------------------------------
// Initialization & Event Listeners
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  // Navigation Tabs
  const navButtons = document.querySelectorAll('.nav-item');
  const panels = document.querySelectorAll('.panel-section');

  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.dataset.tab;

      navButtons.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      document.getElementById(`tab-${tabId}`).classList.add('active');

      // Trigger plot resize / updates for newly visible tab
      if (tabId === 'overview') renderOverviewPreview(document.getElementById('overview-preview-select').value);
      if (tabId === 'patterns') updatePatternPanel();
      if (tabId === 'scaling') renderScalingTab();
      if (tabId === 'accuracy') renderTaskAccuracyTab();
      if (tabId === 'depth') renderDepthStudyTab();
      if (tabId === 'exports') renderExportsInspector();
    });
  });

  // Overview Preview Change
  document.getElementById('overview-preview-select').addEventListener('change', (e) => {
    renderOverviewPreview(e.target.value);
  });

  // Pattern Controls Change
  document.getElementById('pattern-seq-len').addEventListener('input', updatePatternPanel);
  document.getElementById('pattern-win-size').addEventListener('input', updatePatternPanel);
  document.getElementById('pattern-variant').addEventListener('change', updatePatternPanel);

  // Scaling Measure Radio Change
  document.querySelectorAll('input[name="scaling-measure"]').forEach(r => {
    r.addEventListener('change', renderScalingTab);
  });

  // Accuracy Task Radio Change
  document.querySelectorAll('input[name="accuracy-task"]').forEach(r => {
    r.addEventListener('change', renderTaskAccuracyTab);
  });

  // CSV Download Buttons
  document.getElementById('download-mask-csv').addEventListener('click', downloadMaskCSV);
  document.getElementById('download-sweep-csv').addEventListener('click', downloadSweepCSV);
  document.getElementById('download-copy-csv').addEventListener('click', downloadCopyCSV);
  document.getElementById('download-classification-csv').addEventListener('click', downloadClassificationCSV);
  document.getElementById('download-depth-csv').addEventListener('click', downloadDepthCSV);

  // Initial renders
  renderOverviewPreview('Overlapping sliding-window');
  updatePatternPanel();
  renderScalingTab();
  renderTaskAccuracyTab();
  renderDepthStudyTab();
  renderExportsInspector();
});
