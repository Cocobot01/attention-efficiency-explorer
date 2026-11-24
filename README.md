# Attention Efficiency Explorer

**Beyond Quadratic: Exploring Efficient Variants of Transformer Attention**
MSc Data Science & AI dissertation, University of Liverpool · Distinction (73%) · Shashidhar Kumar

Standard Transformer attention scales quadratically with sequence length. This project implements four attention variants in PyTorch, benchmarks their runtime, memory and accuracy, and presents the results in an interactive Streamlit dashboard.

## Key results (sequence length 8,192)

| Variant | Peak memory | Runtime | vs standard |
|---|---|---|---|
| Standard (full) attention | 8,243 MB | 147.0 ms | reference |
| Naive sliding-window (masked) | 8,819 MB | 216.4 ms | no saving |
| Disjoint block | 55 MB | 1.3 ms | 151× lighter |
| Overlapping sliding-window | 95 MB | 2.2 ms | 87× lighter |

- **Masking alone saves nothing.** The naive variant still builds the full score matrix, so it is slower than standard attention.
- **Accuracy holds on local tasks.** Block-local attention matches standard attention on long-document classification (99.0% vs 98.5%, arXiv vs PubMed).
- **Long-range tasks expose the limit.** On a 128-token copy task, windowed variants stay at chance whenever the window is smaller than the gap.

![Efficiency scaling](screenshots/efficiency_scaling.png)
![Task accuracy](screenshots/task_accuracy.png)
![Attention patterns](screenshots/attention_patterns.png)

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501. On Windows, if `streamlit` isn't recognised, use `python -m streamlit run app.py`. Tested on Streamlit 1.62.

## Panels

| Panel | What it shows | Source |
|---|---|---|
| Overview | The four variants and headline cost figures at 8,192 tokens | `four_way_sweep_results.json` |
| Attention patterns | Live attention mask for any variant, sequence length and window size | computed live |
| Efficiency scaling | Runtime and peak memory (log–log) for all four variants | `four_way_sweep_results.json` |
| Task accuracy | Copy task by window size, or text classification | `copy_task_v2_results.json`, `classification_results.json` |
| Depth study | Copy accuracy against layer count, with a standard-attention control | `depth_results.json` |
| Data & exports | Every source file, inspectable and downloadable | `data/` |

All results are read from JSON files in `data/`, written directly by the experiment notebook. No result is hard-coded.

**Exports:** CSV tables (the "Measured values" expander on each panel), PNG charts (camera icon in the chart toolbar), raw JSON (Data & exports panel), and attention masks as CSV.

## Interpretation choices

- **Naming:** the non-overlapping variant is called *disjoint block* throughout, not "sliding-window", because tokens never attend across block boundaries.
- **Degenerate case:** the w=128 copy-task result is shaded and annotated. At that setting only three blocks exist, so it is effectively full attention and not evidence that local attention solves long-range tasks.
- **Classification scope:** the classification result reflects how local the task is. The overlapping variant has not yet been evaluated on it.

## Notes on the data files

`depth_results.json` includes both the overlapping-window results and the standard-attention control (`standard_by_depth`), which is needed to interpret the depth study. To refresh a panel, replace the matching file in `data/` and reload. Older key names (`true_sliding_window`, `overlapping_w32`) are still accepted.

## Licence

MIT
