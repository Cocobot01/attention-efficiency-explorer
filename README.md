# Attention Efficiency Explorer

Interactive results dashboard for **Beyond Quadratic: Exploring Efficient
Variants of Transformer Attention** (COMP702, Shashidhar Kumar, 201995936).

Every panel reads from the JSON files in `data/`, written directly by the
experiment notebook. No result is hard-coded.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. On Windows, if `streamlit` isn't recognised,
use `python -m streamlit run app.py`.

Tested end to end: all six panels and every control branch render without
errors, on Streamlit 1.62.

## Panels

| Panel | What it shows | Source |
|---|---|---|
| Overview | The four variants, headline cost figures at 8,192 tokens | `four_way_sweep_results.json` |
| Attention patterns | Live-generated attention mask for any variant, sequence length and window size | computed, no file needed |
| Efficiency scaling | Runtime and peak memory, log–log, four variants | `four_way_sweep_results.json` |
| Task accuracy | Copy task by window size, or text classification | `copy_task_v2_results.json`, `classification_results.json` |
| Depth study | Copy accuracy against layer count, with the standard-attention control | `depth_results.json` |
| Data & exports | Every source file, inspectable and downloadable | all of `data/` |

## Exports

- **Tables** — each panel has a *Measured values* expander with a CSV download.
- **Charts** — hover any chart and use the camera icon in its toolbar to save a PNG.
- **Raw results** — the Data & exports panel serves every JSON file.
- **Masks** — the Attention patterns panel exports the generated mask as CSV.

## Data files

`depth_results.json` here is the version containing **both** the overlapping
results and the `standard_by_depth` control (originally exported as
`depth_results_2.json`). The control line is what makes the depth result
interpretable, so don't overwrite this with the earlier single-series file.

To refresh any panel, drop a newer export into `data/` under the same filename
and reload the page. The loaders also accept the older key names
(`true_sliding_window`, `overlapping_w32`), so previous exports still work.

## Framing carried through the interface

Three points your supervisor raised are built into the panels rather than left
to narration:

- The disjoint variant is called **disjoint block** everywhere, never "true
  sliding-window".
- The w=128 copy-task result is shaded **degenerate** and annotated, because
  only three blocks exist at that setting, making it effectively full attention.
- The classification result is framed as a **property of the task**, with a note
  that the overlapping variant has not yet been run on it.
