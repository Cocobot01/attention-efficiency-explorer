"""
Attention Efficiency Explorer
Beyond Quadratic: Exploring Efficient Variants of Transformer Attention

COMP702 MSc Dissertation — Shashidhar Kumar (201995936)
University of Liverpool

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Attention Efficiency Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

INK = "#14162B"
MUTED = "#6B7089"
HAIRLINE = "#E4E5EC"
ACCENT = "#4338CA"

# Variant colours are consistent across every chart, so a colour means the
# same thing wherever the reader meets it.
COLOR = {
    "Standard (full) attention": INK,
    "Naive sliding-window (masked)": "#B45309",
    "Disjoint block": "#0F766E",
    "Overlapping sliding-window": ACCENT,
}

VARIANT_LABELS = {
    "standard": "Standard (full) attention",
    "naive_sliding_window": "Naive sliding-window (masked)",
    "disjoint_block": "Disjoint block",
    "overlapping_sliding_window": "Overlapping sliding-window",
}

PLOT_FONT = dict(family="IBM Plex Sans, sans-serif", size=13, color=INK)

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap" rel="stylesheet">
<style>
  .stApp { background: #FAFAFC; }
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3 {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    color: #14162B !important;
    letter-spacing: -0.01em;
  }
  h1 { font-weight: 600 !important; line-height: 1.15 !important; }
  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.70rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6B7089;
    margin-bottom: 0.35rem;
  }
  .standfirst {
    color: #6B7089;
    font-size: 0.95rem;
    max-width: 64ch;
    line-height: 1.6;
    margin-bottom: 1.4rem;
  }
  .tile {
    background: #FFFFFF;
    border: 1px solid #E4E5EC;
    border-left: 3px solid #4338CA;
    padding: 0.85rem 1rem;
  }
  .tile-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6B7089;
  }
  .tile-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem;
    font-weight: 600;
    color: #14162B;
    line-height: 1.35;
  }
  .tile-note { font-size: 0.78rem; color: #6B7089; }
  .tile { min-height: 118px; }
  .rule { border-top: 1px solid #E4E5EC; margin: 1.6rem 0 1.2rem 0; }
  .caveat {
    border-left: 3px solid #B45309;
    background: #FFFFFF;
    border-top: 1px solid #E4E5EC;
    border-right: 1px solid #E4E5EC;
    border-bottom: 1px solid #E4E5EC;
    padding: 0.85rem 1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #14162B;
  }
  section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E4E5EC;
  }
  .sidebar-title {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #14162B;
  }
  .sidebar-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #6B7089;
  }
  .stDownloadButton button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    border-radius: 0;
    border: 1px solid #14162B;
    color: #14162B;
    background: #FFFFFF;
  }
  .stDownloadButton button:hover {
    background: #14162B; color: #FFFFFF; border-color: #14162B;
  }
</style>
""",
    unsafe_allow_html=True,
)


def eyebrow(text):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def standfirst(text):
    st.markdown(f'<div class="standfirst">{text}</div>', unsafe_allow_html=True)


def tile(col, label, value, note=""):
    col.markdown(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div>'
        f'<div class="tile-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def caveat(html):
    st.markdown(f'<div class="caveat">{html}</div>', unsafe_allow_html=True)


def rule():
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


def style_fig(fig, height=420, legend=True):
    fig.update_layout(
        font=PLOT_FONT,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=10, r=10, t=52 if fig.layout.title.text else 12, b=10),
        showlegend=legend,
    )
    if fig.layout.title.text:
        fig.update_layout(title=dict(
            font=dict(family="IBM Plex Serif, serif", size=15, color=INK), x=0))
    if legend:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left",
                        x=0, title_text="", font=dict(size=11.5)),
            margin=dict(l=10, r=10, t=52 if fig.layout.title.text else 12, b=70),
        )
    fig.update_xaxes(gridcolor=HAIRLINE, zeroline=False, linecolor=HAIRLINE,
                     tickfont=dict(family="IBM Plex Mono, monospace", size=11))
    fig.update_yaxes(gridcolor=HAIRLINE, zeroline=False, linecolor=HAIRLINE,
                     tickfont=dict(family="IBM Plex Mono, monospace", size=11))
    return fig


def download_csv(df, filename, label="Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data
def load_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def first_key(d, candidates):
    for c in candidates:
        if d and c in d:
            return c
    return None


def missing(filename, what):
    caveat(
        f"<b>{what} has no data file.</b> Add <code>{filename}</code> to the "
        f"<code>data/</code> folder to populate this panel. Expected keys are "
        f"listed in README.md."
    )


@st.cache_data
def sweep_dataframe():
    """Long-format frame of runtime and memory by variant and sequence length."""
    four = load_json("four_way_sweep_results.json")
    three = load_json("full_sweep_results.json")
    source = four if four else three
    if source is None:
        return pd.DataFrame()

    key_for = {
        "standard": "standard",
        "naive_sliding_window": "naive_sliding_window",
        "disjoint_block": first_key(source, ["disjoint_block", "true_sliding_window"]),
        "overlapping_sliding_window": first_key(
            source, ["overlapping_sliding_window", "overlapping"]),
    }
    rows = []
    for label_key, src_key in key_for.items():
        if not src_key or src_key not in source:
            continue
        for p in source[src_key]:
            rows.append({
                "variant": VARIANT_LABELS[label_key],
                "seq_len": p["seq_len"],
                "time_s": p.get("time_mean", p.get("time_sec")),
                "mem_MB": p.get("mem_mean", p.get("peak_mem_MB")),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mask generator — the mechanism itself, computed live
# ---------------------------------------------------------------------------


def generate_mask(variant, seq_len, window_size):
    if variant == "Standard (full) attention":
        return np.ones((seq_len, seq_len), dtype=bool)

    num_blocks = int(np.ceil(seq_len / window_size))
    mask = np.zeros((seq_len, seq_len), dtype=bool)
    for b in range(num_blocks):
        q0, q1 = b * window_size, min((b + 1) * window_size, seq_len)
        if variant == "Disjoint block":
            blocks = [b]
        else:
            blocks = [x for x in (b - 1, b, b + 1) if 0 <= x < num_blocks]
        for ab in blocks:
            k0, k1 = ab * window_size, min((ab + 1) * window_size, seq_len)
            mask[q0:q1, k0:k1] = True
    return mask


def mask_figure(mask, height=520, title=None):
    fig = px.imshow(
        mask.astype(int),
        color_continuous_scale=[[0, "#F1F1F6"], [1, ACCENT]],
        labels=dict(x="Key position", y="Query position"),
        aspect="equal",
    )
    fig.update_layout(coloraxis_showscale=False)
    if title:
        fig.update_layout(title=title)
    return style_fig(fig, height=height, legend=False)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown('<div class="sidebar-title">Attention Efficiency Explorer</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">COMP702 &middot; 201995936</div>',
                    unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Panel",
    ["Overview", "Attention patterns", "Efficiency scaling",
     "Task accuracy", "Depth study", "Data & exports"],
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-sub">Ethics</div>', unsafe_allow_html=True)
st.sidebar.caption(
    "Data category A, participant category 0. No human- or animal-derived data; "
    "all sequences are synthetic or drawn from openly licensed benchmarks."
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    eyebrow("MSc dissertation &middot; University of Liverpool")
    st.title("Beyond Quadratic")
    standfirst(
        "Standard self-attention compares every query position with every key "
        "position, so its cost grows with the square of sequence length. This "
        "explorer walks through four attention variants and separates the ones "
        "that genuinely reduce that cost from the one that only appears to."
    )

    df = sweep_dataframe()
    if not df.empty:
        longest = int(df["seq_len"].max())
        at_max = df[df["seq_len"] == longest].set_index("variant")
        std_t = at_max.loc["Standard (full) attention", "time_s"]
        std_m = at_max.loc["Standard (full) attention", "mem_MB"]
        c = st.columns(4)
        tile(c[0], "Longest sequence", f"{longest:,}", "tokens profiled")
        tile(c[1], "Standard attention", f"{std_m/1024:.1f} GB",
             f"{std_t*1000:.0f} ms per forward pass")
        if "Overlapping sliding-window" in at_max.index:
            ov_t = at_max.loc["Overlapping sliding-window", "time_s"]
            ov_m = at_max.loc["Overlapping sliding-window", "mem_MB"]
            tile(c[2], "Overlapping window", f"{ov_m:.0f} MB",
                 f"{ov_t*1000:.1f} ms per forward pass")
            tile(c[3], "Difference", f"{std_m/ov_m:.0f}\u00d7 lighter",
                 f"{std_t/ov_t:.0f}\u00d7 faster at this length")

    rule()
    eyebrow("The four variants")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            """
**Standard (full) attention** — every position attends to every other.
Quadratic in time and memory, and the reference point for accuracy.

**Naive sliding-window (masked)** — builds the full score matrix, then masks it
down to a local window. It reproduces the *accuracy* of a local method without
any of the saving: the expensive step still happens, so it lands slightly
slower than standard attention.

**Disjoint block** — the sequence is cut into non-overlapping blocks and each
position attends only inside its own block. Genuinely linear, but information
never crosses a block boundary, not even into the adjacent block.

**Overlapping sliding-window** — each block additionally attends to its
immediate left and right neighbours, in the manner of Longformer. Still linear,
with roughly three times the scores per query and a receptive field that widens
with depth.
            """
        )
    with right:
        demo = st.selectbox(
            "Preview a variant",
            ["Overlapping sliding-window", "Disjoint block",
             "Standard (full) attention"],
        )
        st.plotly_chart(mask_figure(generate_mask(demo, 48, 8), height=330),
                        width="stretch")
        st.caption(
            "Attended query–key pairs at sequence length 48, window 8. "
            "The Attention patterns panel makes these controls live."
        )

# ---------------------------------------------------------------------------
# Attention patterns
# ---------------------------------------------------------------------------

elif page == "Attention patterns":
    eyebrow("Panel 01 &middot; Mechanism")
    st.title("Attention patterns")
    standfirst(
        "This panel computes the mask each variant applies, live, from the same "
        "block logic used in the implementation. Nothing here is stored output — "
        "move a control and the mask is rebuilt."
    )

    c1, c2, c3 = st.columns(3)
    seq_len = c1.slider("Sequence length", 16, 256, 64, step=16)
    window_size = c2.slider("Window size", 2, 64, 8, step=2)
    variant = c3.selectbox(
        "Variant",
        ["Overlapping sliding-window", "Disjoint block", "Standard (full) attention"],
    )

    mask = generate_mask(variant, seq_len, window_size)
    n_blocks = int(np.ceil(seq_len / window_size))
    coverage = mask.sum() / mask.size * 100

    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(mask_figure(mask, height=520), width="stretch")
    with right:
        tile(st, "Attended pairs", f"{coverage:.1f}%",
             f"{int(mask.sum()):,} of {mask.size:,} query–key pairs")
        st.write("")
        tile(st, "Blocks", f"{n_blocks}",
             "sequence length \u00f7 window size, rounded up")
        st.write("")
        if variant != "Standard (full) attention":
            reach = window_size * (2 if variant == "Overlapping sliding-window" else 1)
            tile(st, "Reach per layer", f"{reach} tokens",
                 "furthest a token's influence travels in one layer")
        st.write("")
        st.download_button(
            "Download mask as CSV",
            data=pd.DataFrame(mask.astype(int)).to_csv(index=False).encode(),
            file_name=f"mask_n{seq_len}_w{window_size}.csv",
            mime="text/csv",
        )

    if variant == "Overlapping sliding-window" and n_blocks <= 3:
        st.write("")
        caveat(
            "<b>Degenerate configuration.</b> With three blocks or fewer, every "
            "block already reaches every other block, so this is full attention in "
            "all but name. Accuracy measured here says nothing about local "
            "attention on long sequences — the same effect explains the w=128 "
            "result in the Task accuracy panel."
        )

# ---------------------------------------------------------------------------
# Efficiency scaling
# ---------------------------------------------------------------------------

elif page == "Efficiency scaling":
    eyebrow("Panel 02 &middot; Cost")
    st.title("Efficiency scaling")
    standfirst(
        "Forward-pass runtime and peak memory, measured within a single GPU "
        "session across sequence lengths from 128 to 8,192."
    )

    df = sweep_dataframe()
    if df.empty:
        missing("four_way_sweep_results.json", "Efficiency scaling")
    else:
        measure = st.radio("Measure", ["Both", "Runtime", "Peak memory"],
                           horizontal=True)

        def line(y, title, ylab):
            fig = px.line(
                df, x="seq_len", y=y, color="variant", markers=True,
                log_x=True, log_y=True, title=title, color_discrete_map=COLOR,
                labels={"seq_len": "Sequence length", y: ylab},
            )
            return style_fig(fig)

        if measure == "Both":
            a, b = st.columns(2)
            a.plotly_chart(line("time_s", "Runtime", "Seconds"), width="stretch")
            b.plotly_chart(line("mem_MB", "Peak memory", "MB"), width="stretch")
        elif measure == "Runtime":
            st.plotly_chart(line("time_s", "Runtime", "Seconds"), width="stretch")
        else:
            st.plotly_chart(line("mem_MB", "Peak memory", "MB"), width="stretch")

        longest = int(df["seq_len"].max())
        at_max = df[df["seq_len"] == longest].set_index("variant")
        std_m = at_max.loc["Standard (full) attention", "mem_MB"]
        std_t = at_max.loc["Standard (full) attention", "time_s"]

        rule()
        eyebrow(f"At sequence length {longest:,}")
        cols = st.columns(len(at_max))
        for col, (name, row) in zip(cols, at_max.iterrows()):
            ratio = std_m / row["mem_MB"]
            if name.startswith("Standard"):
                note = "reference"
            elif ratio < 1.05:
                # Rounding a penalty into a "1x saving" would misreport the result.
                note = "no saving over standard"
            else:
                note = f"{ratio:.0f}\u00d7 lighter than standard"
            tile(col, name, f"{row['mem_MB']:.0f} MB",
                 f"{row['time_s']*1000:.1f} ms &middot; {note}")

        st.write("")
        naive_t = at_max.loc["Naive sliding-window (masked)", "time_s"]
        caveat(
            f"The naive masked variant is <b>slower</b> than standard attention "
            f"({naive_t*1000:.0f} ms against {std_t*1000:.0f} ms) because masking "
            f"happens after the full score matrix has been built. Restricting what "
            f"a model attends to only saves compute if the restriction changes what "
            f"gets computed."
        )
        with st.expander("Measured values"):
            st.dataframe(df, width="stretch")
            download_csv(df, "efficiency_sweep.csv")

# ---------------------------------------------------------------------------
# Task accuracy
# ---------------------------------------------------------------------------

elif page == "Task accuracy":
    eyebrow("Panel 03 &middot; Capability")
    st.title("Task accuracy")

    task = st.radio("Task", ["Sequence copy (long-range)", "Text classification"],
                    horizontal=True)

    if task == "Sequence copy (long-range)":
        data = load_json("copy_task_v2_results.json")
        if data is None:
            missing("copy_task_v2_results.json", "Copy task")
        else:
            seq_len = data.get("seq_len", 0)
            standard_acc = data.get("standard_accuracy")
            ov_key = first_key(data, ["overlapping_accuracy", "overlapping"])
            dj_key = first_key(data, ["disjoint_accuracy", "disjoint_block_accuracy"])

            standfirst(
                f"The model must reproduce a sequence of length {seq_len} presented "
                f"earlier in the input, a copy distance of about {int(seq_len)+1} "
                f"tokens. Success requires carrying information across that gap."
            )

            rows = []
            if ov_key:
                rows += [{"window": int(w), "accuracy": a,
                          "variant": "Overlapping sliding-window"}
                         for w, a in data[ov_key].items()]
            if dj_key:
                rows += [{"window": int(w), "accuracy": a, "variant": "Disjoint block"}
                         for w, a in data[dj_key].items()]
            df = pd.DataFrame(rows).sort_values(["variant", "window"])

            fig = px.bar(
                df, x="window", y="accuracy", color="variant", barmode="group",
                color_discrete_map=COLOR, title="Copy accuracy by window size",
                labels={"window": "Window size", "accuracy": "Accuracy"},
            )
            fig.update_xaxes(type="category")
            if standard_acc is not None:
                fig.add_hline(
                    y=standard_acc, line_dash="dash", line_color=INK, line_width=1.5,
                    annotation_text=f"Standard attention · {standard_acc:.2f}",
                    annotation_position="top left",
                    annotation_font=dict(family="IBM Plex Mono", size=11))
            fig.add_hline(
                y=0.02, line_dash="dot", line_color=MUTED, line_width=1,
                annotation_text="Chance", annotation_position="bottom left",
                annotation_font=dict(family="IBM Plex Mono", size=11))
            n_cat = df["window"].nunique()
            fig.add_vrect(
                x0=n_cat - 1.5, x1=n_cat - 0.5, fillcolor=MUTED, opacity=0.10,
                line_width=0, annotation_text="degenerate",
                annotation_position="top right",
                annotation_font=dict(family="IBM Plex Mono", size=11, color=MUTED))
            fig.update_yaxes(range=[0, 1.08])
            st.plotly_chart(style_fig(fig, height=440), width="stretch")

            max_w = int(df["window"].max())
            top = data[ov_key][str(max_w)]
            caveat(
                f"<b>Two results, read separately.</b> At every window size a long "
                f"sequence would actually use, both variants sit at chance: a local "
                f"window cannot route information across a gap wider than its reach. "
                f"At w={max_w} the overlapping variant reaches {top:.2f}, but only "
                f"three blocks exist at that setting, so every block sees every "
                f"other and this is effectively full attention. It is evidence that "
                f"the overlap mechanism works — the disjoint variant, which never "
                f"crosses a block boundary, stays at chance in the identical "
                f"configuration — and not evidence that local attention solves "
                f"long-range tasks."
            )
            with st.expander("Measured values"):
                st.dataframe(df, width="stretch")
                download_csv(df, "copy_task_accuracy.csv")

    else:
        data = load_json("classification_results.json")
        if data is None:
            missing("classification_results.json", "Classification")
        else:
            standfirst(
                "A short-sequence text classification task, included to test whether "
                "the copy-task failure carries over to a more realistic workload."
            )
            pretty = {
                "standard_accuracy": "Standard (full) attention",
                "sliding_window_w32_accuracy": "Disjoint block, w=32",
                "sliding_window_w64_accuracy": "Disjoint block, w=64",
            }
            df = pd.DataFrame([{"configuration": pretty.get(k, k), "accuracy": v}
                               for k, v in data.items()])
            fig = px.bar(df, x="configuration", y="accuracy", text_auto=".3f",
                         labels={"configuration": "", "accuracy": "Accuracy"},
                         title="Classification accuracy")
            fig.update_traces(
                marker_color=[INK, COLOR["Disjoint block"], COLOR["Disjoint block"]],
                textfont=dict(family="IBM Plex Mono", size=12))
            fig.update_yaxes(range=[0, 1.08])
            st.plotly_chart(style_fig(fig, height=400, legend=False), width="stretch")

            caveat(
                "<b>Read this as a property of the task, not of the model.</b> Local "
                "and full attention scoring alike here indicates the task can be "
                "solved from local evidence, so it never exercises the long-range "
                "capability the copy task isolates. It is not evidence that local "
                "attention substitutes for full attention in general. These runs "
                "also cover standard and disjoint block attention only; the "
                "overlapping variant has not yet been evaluated on this task."
            )
            with st.expander("Measured values"):
                st.dataframe(df, width="stretch")
                download_csv(df, "classification_accuracy.csv")

# ---------------------------------------------------------------------------
# Depth study
# ---------------------------------------------------------------------------

elif page == "Depth study":
    eyebrow("Panel 04 &middot; Control")
    st.title("Depth study")
    standfirst(
        "Stacking local-attention layers widens the receptive field in theory, "
        "since each layer carries information one window further. This panel asks "
        "whether that recovers the copy task in practice."
    )

    data = load_json("depth_results.json")
    if data is None:
        missing("depth_results.json", "Depth study")
    else:
        ov_key = first_key(data, ["overlapping_w32_by_depth", "overlapping_w32"])
        std_key = first_key(data, ["standard_by_depth"])
        rows = []
        if ov_key:
            rows += [{"depth": int(d), "accuracy": a,
                      "variant": "Overlapping sliding-window"}
                     for d, a in data[ov_key].items()]
        if std_key:
            rows += [{"depth": int(d), "accuracy": a,
                      "variant": "Standard (full) attention"}
                     for d, a in data[std_key].items()]
        df = pd.DataFrame(rows).sort_values(["variant", "depth"])

        fig = px.line(
            df, x="depth", y="accuracy", color="variant", markers=True,
            color_discrete_map=COLOR,
            title="Copy accuracy against number of layers",
            labels={"depth": "Attention layers", "accuracy": "Accuracy"},
        )
        fig.add_hline(y=0.02, line_dash="dot", line_color=MUTED, line_width=1,
                      annotation_text="Chance", annotation_position="bottom left",
                      annotation_font=dict(family="IBM Plex Mono", size=11))
        fig.update_yaxes(range=[0, 1.08])
        fig.update_xaxes(dtick=1)
        st.plotly_chart(style_fig(fig, height=440), width="stretch")

        if std_key and ov_key:
            ov_vals = list(data[ov_key].values())
            std_vals = list(data[std_key].values())
            c = st.columns(2)
            tile(c[0], "Overlapping, w=32",
                 f"{min(ov_vals):.3f}\u2013{max(ov_vals):.3f}", "across 1\u20134 layers")
            tile(c[1], "Standard control",
                 f"{min(std_vals):.3f}\u2013{max(std_vals):.3f}", "across 1\u20134 layers")

        st.write("")
        caveat(
            "<b>The control is what makes this readable.</b> Standard attention "
            "holds high accuracy at every depth, so the residual connections, "
            "learning rate and training loop are sound at depth; a flat "
            "local-attention curve therefore reflects the architecture rather than "
            "a broken setup. The claim stays bounded: depth did not compensate "
            "<i>within the range tested</i>. Four layers at w=32 gives a "
            "theoretical reach well short of the copy distance, so this does not "
            "show that depth could never help."
        )
        with st.expander("Measured values"):
            st.dataframe(df, width="stretch")
            download_csv(df, "depth_study.csv")

# ---------------------------------------------------------------------------
# Data & exports
# ---------------------------------------------------------------------------

elif page == "Data & exports":
    eyebrow("Panel 05 &middot; Provenance")
    st.title("Data & exports")
    standfirst(
        "Every figure in this explorer is read from the files below, written "
        "directly by the experiment notebook. No result is hard-coded."
    )

    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        st.write("No result files found in the data folder.")
    for f in files:
        st.markdown(f"**`{f.name}`** &middot; {f.stat().st_size:,} bytes")
        with st.expander("Inspect"):
            st.json(json.loads(f.read_text()), expanded=False)
        st.download_button(f"Download {f.name}", data=f.read_bytes(),
                           file_name=f.name, mime="application/json",
                           key=f"dl_{f.name}")
        st.write("")

    rule()
    eyebrow("Exporting figures")
    st.markdown(
        "Every chart carries a toolbar in its top-right corner on hover. The "
        "camera icon saves that chart as a PNG. Tabular exports sit under the "
        "*Measured values* expander on each panel."
    )
