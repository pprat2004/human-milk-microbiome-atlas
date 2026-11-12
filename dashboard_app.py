"""
Human Milk Microbiome Atlas
===========================
An interactive research dashboard built on the MAMI and PRIME human-milk
microbiome resources.

Run locally:
    pip install -r requirements.txt
    streamlit run dashboard_app.py

The Research Assistant page uses Google Gemini (embeddings + answer generation) to
answer questions from a curated literature corpus, with source citations.

Layout of this file
-------------------
1. Page setup and visual theme (colours, CSS)
2. Data loading helpers (cached for speed)
3. Small UI building blocks (hero banner, insight cards, section headers)
4. One function per dashboard page
5. Sidebar navigation and page routing
"""

import json
import os
from pathlib import Path

try:  # optional: load GEMINI_API_KEY from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rag_engine import (
    CHAT_MODEL_CHOICES,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_CHOICES,
    ask_rag,
    build_index,
    load_corpus,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE SETUP AND THEME
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"

# One palette used everywhere so every chart feels part of the same family.
TEAL = "#0F766E"
TEAL_LIGHT = "#5EEAD4"
CORAL = "#F97316"
NAVY = "#0F2A43"
SLATE = "#475569"
MIST = "#F1F7F7"

# Categorical colours for the seven MAMI studies (colour-blind friendly mix).
STUDY_COLORS = [
    "#0F766E", "#F97316", "#2563EB", "#9333EA",
    "#CA8A04", "#DB2777", "#16A34A",
]

st.set_page_config(
    page_title="Human Milk Microbiome Atlas",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
/* Overall spacing */
.block-container {{padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1300px;}}

/* Hero banner */
.hero {{
    padding: 34px 38px; border-radius: 22px; margin-bottom: 22px; color: white;
    background: radial-gradient(circle at 85% 15%, rgba(94,234,212,.35), transparent 45%),
                linear-gradient(120deg, {NAVY} 0%, #134E5E 55%, {TEAL} 100%);
    box-shadow: 0 10px 30px rgba(15,42,67,.18);
}}
.hero h1 {{margin: 0 0 8px 0; font-size: 2.35rem; letter-spacing: -0.5px; color: white;}}
.hero p {{margin: 0; font-size: 1.05rem; color: #d8f3ef; max-width: 820px; line-height: 1.55;}}
.hero .pill {{
    display: inline-block; padding: 4px 12px; margin: 14px 8px 0 0; border-radius: 999px;
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
    font-size: .82rem; color: #ecfeff;
}}

/* KPI tiles (st.metric) */
[data-testid="stMetric"] {{
    background: white; border: 1px solid #e2ecec; border-left: 5px solid {TEAL};
    padding: 16px 18px; border-radius: 14px; box-shadow: 0 2px 10px rgba(15,42,67,.05);
}}
[data-testid="stMetricLabel"] {{color: {SLATE}; font-weight: 600;}}
[data-testid="stMetricValue"] {{color: {NAVY};}}

/* Insight cards */
.card {{
    background: white; border: 1px solid #e2ecec; border-radius: 16px; padding: 18px 20px;
    height: 100%; box-shadow: 0 2px 10px rgba(15,42,67,.05);
}}
.card h4 {{margin: 0 0 6px 0; color: {NAVY}; font-size: 1.02rem;}}
.card p {{margin: 0; color: {SLATE}; font-size: .93rem; line-height: 1.5;}}
.card .icon {{font-size: 1.6rem; margin-bottom: 6px;}}

/* "How to read this" explainer box */
.explain {{
    background: {MIST}; border-left: 5px solid {TEAL}; border-radius: 10px;
    padding: 14px 18px; margin: 6px 0 18px 0; color: #1f3b4d; font-size: .95rem; line-height: 1.55;
}}
.explain b {{color: {TEAL};}}

/* Section header with accent underline */
.section-title {{
    font-size: 1.35rem; font-weight: 700; color: {NAVY}; margin: 26px 0 4px 0;
    padding-bottom: 6px; border-bottom: 3px solid {TEAL_LIGHT}; display: inline-block;
}}
.section-sub {{color: {SLATE}; margin-bottom: 12px; font-size: .95rem;}}

/* Sidebar */
[data-testid="stSidebar"] {{background: linear-gradient(180deg, #0F2A43 0%, #134E5E 100%);}}
[data-testid="stSidebar"] * {{color: #e6f7f5 !important;}}
[data-testid="stSidebar"] hr {{border-color: rgba(255,255,255,.2);}}

/* Retrieved-source cards in the assistant */
.source {{
    border: 1px solid #e2ecec; border-left: 5px solid {CORAL}; border-radius: 12px;
    padding: 10px 14px; margin-bottom: 8px; background: #fffaf5;
}}
.source b {{color: {NAVY};}}
.source .meta {{color: {SLATE}; font-size: .85rem;}}
.tag {{display:inline-block; padding:1px 9px; border-radius:999px; font-size:.75rem; margin-left:6px;
      background:#e0f2f1; color:{TEAL};}}
.tag.internal {{background:#ffedd5; color:#c2410c;}}

/* Footer */
.footer {{text-align: center; color: #7b8794; font-size: .85rem; margin-top: 36px;}}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING (cached so pages switch instantly)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    """Read one CSV from the data folder; return an empty frame if absent."""
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def show_image(name: str, caption: str | None = None) -> None:
    """Display a pre-rendered figure from the data folder."""
    path = DATA / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


alpha = load_csv("mami_sample_alpha_diversity.csv")
alpha_study = load_csv("mami_alpha_diversity_by_study.csv")
clr_assoc = load_csv("mami_clr_feature_associations_all_pseudocounts.csv")
robust = load_csv("mami_clr_feature_robustness.csv")
ml_metrics = load_csv("mami_study_classifier_cv_metrics.csv")
class_metrics = load_csv("mami_study_classifier_per_class.csv")
study_country = load_csv("mami_study_country_cross_tab.csv")
inventory = load_csv("dataset_coverage_inventory.csv")
source_catalog = load_csv("rag_source_catalog.csv")

# Consistent colour for each study across every chart.
STUDIES = sorted(alpha["Study"].unique()) if not alpha.empty else []
STUDY_COLOR_MAP = {s: STUDY_COLORS[i % len(STUDY_COLORS)] for i, s in enumerate(STUDIES)}

# Friendly names for alpha-diversity metrics.
METRICS = {
    "Observed_richness": ("Observed richness", "How many different microbial genera are detected in a sample."),
    "Shannon": ("Shannon diversity", "Combines richness and evenness: higher means a more varied, balanced community."),
    "Simpson_1_minus_D": ("Simpson diversity", "The chance that two randomly drawn microbes belong to different genera."),
    "Pielou_evenness": ("Pielou evenness", "How evenly microbes share the community (1 = perfectly even)."),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMALL UI BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────
def hero(title: str, subtitle: str, pills: list[str]) -> None:
    pill_html = "".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{pill_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def card(icon: str, title: str, text: str) -> str:
    return f'<div class="card"><div class="icon">{icon}</div><h4>{title}</h4><p>{text}</p></div>'


def explain(text: str) -> None:
    """A short plain-language 'how to read this chart' box."""
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 460) -> go.Figure:
    """Apply one consistent, clean look to every Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=NAVY),
        title=dict(font=dict(size=17, color=NAVY)),
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(bgcolor="rgba(255,255,255,0.7)"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_home() -> None:
    n_samples = len(alpha) if not alpha.empty else 1128
    n_studies = alpha["Study"].nunique() if not alpha.empty else 7
    n_countries = alpha["Country"].nunique() if not alpha.empty else 6
    n_genera = clr_assoc["feature"].nunique() if not clr_assoc.empty else 0

    hero(
        "Human Milk Microbiome Atlas",
        "Explore the microbial communities of human breast milk across six countries, "
        "seven studies and more than a thousand samples, with interactive diversity, "
        "composition and machine-learning views.",
        ["🧬 MAMI + PRIME", "🌍 6 countries", "📈 Diversity & composition", "🤖 Machine learning", "💬 Gemini research assistant"],
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Milk samples", f"{n_samples:,}")
    c2.metric("Studies", n_studies)
    c3.metric("Countries", n_countries)
    c4.metric("Genera profiled", n_genera)
    c5.metric("PRIME records", "176")

    section("What you can explore", "Each page tells one part of the story. Start anywhere.")
    cols = st.columns(5)
    cols[0].markdown(card("🌍", "Diversity across populations",
                          "Compare richness, Shannon and Simpson diversity between studies and countries."),
                     unsafe_allow_html=True)
    cols[1].markdown(card("🦠", "Microbial signatures",
                          "Find the genera that define each study, ranked by effect size and statistical strength."),
                     unsafe_allow_html=True)
    cols[2].markdown(card("🗺️", "Community map",
                          "See how all 1,128 samples arrange themselves by overall community similarity."),
                     unsafe_allow_html=True)
    cols[3].markdown(card("🤖", "Model insights",
                          "A machine-learning model that recognises a sample's study from its microbes alone."),
                     unsafe_allow_html=True)
    cols[4].markdown(card("💬", "Research assistant",
                          "Ask questions and get cited answers from curated papers, powered by Google Gemini."),
                     unsafe_allow_html=True)

    section("Headline findings")
    if not alpha_study.empty:
        top_rich = alpha_study.sort_values("richness_mean", ascending=False).iloc[0]
        top_shan = alpha_study.sort_values("shannon_mean", ascending=False).iloc[0]
        biggest = alpha_study.sort_values("n", ascending=False).iloc[0]
        f1, f2, f3 = st.columns(3)
        f1.markdown(card("🏆", "Richest communities",
                         f"<b>{top_rich['Study']}</b> ({top_rich['Country']}) shows the highest mean richness, "
                         f"about {top_rich['richness_mean']:.0f} genera per sample."),
                    unsafe_allow_html=True)
        f2.markdown(card("🌈", "Most diverse (Shannon)",
                         f"<b>{top_shan['Study']}</b> ({top_shan['Country']}) leads on Shannon diversity "
                         f"with a mean of {top_shan['shannon_mean']:.2f}."),
                    unsafe_allow_html=True)
        f3.markdown(card("📚", "Largest cohort",
                         f"<b>{biggest['Study']}</b> ({biggest['Country']}) contributes "
                         f"{int(biggest['n']):,} samples, anchoring the atlas."),
                    unsafe_allow_html=True)

    if not ml_metrics.empty:
        best = ml_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
        st.markdown("")
        st.success(
            f"🤖 **Strong microbial fingerprints:** the best model recognises the originating study with "
            f"**{best['accuracy']*100:.1f}% accuracy** (balanced accuracy {best['balanced_accuracy']*100:.1f}%), "
            f"showing that each population carries a distinctive milk-microbiome signature."
        )

    section("Where the data comes from")
    left, right = st.columns(2)
    left.markdown(card("🥛", "MAMI",
                       "A curated collection of human-milk microbiome profiles: 1,128 samples with genus-level "
                       "abundances, study and country labels, and sampling-time information."),
                  unsafe_allow_html=True)
    right.markdown(card("🔬", "PRIME",
                        "A breast-milk metadata resource with 176 records from two projects (USA and France), "
                        "each with participant and BioSample identifiers, all from healthy donors."),
                   unsafe_allow_html=True)


def page_diversity() -> None:
    hero(
        "Diversity across populations",
        "How rich and varied are milk microbial communities, and how do they differ between studies and countries?",
        ["Alpha diversity", "Study profiles", "Country view"],
    )
    if alpha.empty:
        st.info("Add `mami_sample_alpha_diversity.csv` to the data folder to unlock this page.")
        return

    explain(
        "<b>How to read this page.</b> Alpha diversity describes the variety of microbes <i>inside one sample</i>. "
        "Pick a measure below; each box shows the spread of samples in a study, the line is the median, "
        "and every dot is one milk sample. Groups are best read as <b>study-specific population profiles</b>."
    )

    c1, c2 = st.columns([2, 1])
    metric = c1.selectbox(
        "Diversity measure", list(METRICS.keys()), index=1, format_func=lambda k: METRICS[k][0]
    )
    group = c2.radio("Group samples by", ["Study", "Country"], horizontal=True)
    st.caption(f"ℹ️ {METRICS[metric][1]}")

    order = alpha.groupby(group)[metric].median().sort_values(ascending=False).index.tolist()
    fig = px.box(
        alpha, x=group, y=metric, color=group, points="all", category_orders={group: order},
        color_discrete_map=STUDY_COLOR_MAP if group == "Study" else None,
        color_discrete_sequence=STUDY_COLORS,
        title=f"{METRICS[metric][0]} by {group.lower()} (ordered by median)",
        hover_data=[c for c in ["Run", "Study", "Country"] if c in alpha.columns],
    )
    fig.update_traces(marker=dict(size=3, opacity=0.45), jitter=0.5)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=METRICS[metric][0])
    st.plotly_chart(style_fig(fig, 520), use_container_width=True)

    section("Study-level summary", "Average diversity and size of each cohort.")
    if not alpha_study.empty:
        a, b = st.columns(2)
        by_rich = alpha_study.sort_values("richness_mean")
        fig_r = px.bar(
            by_rich, x="richness_mean", y="Study", orientation="h", color="Country",
            text=by_rich["richness_mean"].round(1),
            color_discrete_sequence=STUDY_COLORS, title="Mean observed richness",
        )
        fig_r.update_layout(xaxis_title="Genera per sample", yaxis_title="")
        a.plotly_chart(style_fig(fig_r, 400), use_container_width=True)

        fig_s = px.scatter(
            alpha_study, x="shannon_mean", y="simpson_mean", size="n", color="Country",
            hover_name="Study", size_max=55, color_discrete_sequence=STUDY_COLORS,
            title="Shannon vs Simpson (bubble size = number of samples)",
        )
        fig_s.update_layout(xaxis_title="Mean Shannon diversity", yaxis_title="Mean Simpson diversity")
        b.plotly_chart(style_fig(fig_s, 400), use_container_width=True)

        table = alpha_study.rename(columns={
            "n": "Samples", "richness_mean": "Mean richness", "richness_median": "Median richness",
            "shannon_mean": "Mean Shannon", "shannon_sd": "Shannon SD", "simpson_mean": "Mean Simpson",
        })
        st.dataframe(
            table, use_container_width=True, hide_index=True,
            column_config={
                "Mean richness": st.column_config.ProgressColumn(
                    "Mean richness", format="%.1f", min_value=0, max_value=float(table["Mean richness"].max())),
                "Median richness": st.column_config.NumberColumn(format="%.1f"),
                "Mean Shannon": st.column_config.NumberColumn(format="%.2f"),
                "Shannon SD": st.column_config.NumberColumn(format="%.2f"),
                "Mean Simpson": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    if not study_country.empty:
        section("Who contributed what", "Number of samples contributed by each study in each country.")
        sc = study_country.set_index("Study")
        fig_h = px.imshow(
            sc, text_auto=True, aspect="auto", color_continuous_scale="Teal",
            title="Samples per study and country",
        )
        fig_h.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig_h, 400), use_container_width=True)


def page_signatures() -> None:
    hero(
        "Microbial signatures",
        "Discover which genera characterise each study, and how consistently they do so.",
        ["CLR abundance", "Effect sizes", "Robustness"],
    )
    if clr_assoc.empty:
        st.info("Add the CLR association file to the data folder to unlock this page.")
        return

    explain(
        "<b>How to read this page.</b> Abundances are transformed with the <b>centred log-ratio (CLR)</b>, a standard "
        "method for compositional microbiome data. A <b>higher CLR</b> means a genus is relatively more abundant in a "
        "study. The <b>effect size (ε²)</b> tells you how strongly a genus separates the studies "
        "(larger = more distinctive); the <b>q-value</b> is the statistical significance after multiple-testing correction."
    )

    deltas = sorted(clr_assoc["zero_replacement_delta"].dropna().unique())
    c1, c2 = st.columns(2)
    delta = c1.select_slider(
        "Zero-replacement setting", options=deltas, value=deltas[min(1, len(deltas) - 1)],
        format_func=lambda x: f"{x:g}",
        help="CLR needs non-zero values, so zeros are replaced by a tiny number. Results are shown for three settings.",
    )
    top_n = c2.slider("Genera to show in heatmap", 8, 30, 15)

    work = clr_assoc[clr_assoc["zero_replacement_delta"] == delta].copy()
    work = work.sort_values("epsilon_squared", ascending=False)
    median_cols = [c for c in work.columns if c.startswith("median_CLR_")]

    section("Signature heatmap", "Median CLR of the most distinctive genera in every study.")
    heat = work.head(top_n).set_index("feature")[median_cols]
    heat.columns = [c.replace("median_CLR_", "") for c in heat.columns]
    fig = px.imshow(
        heat, aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        labels=dict(color="Median CLR"), title=f"Top {top_n} most distinctive genera",
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(style_fig(fig, max(420, top_n * 26 + 120)), use_container_width=True)

    section("Strength vs. commonness", "Each bubble is a genus. Up and to the right = common and highly distinctive.")
    work["-log10(q)"] = -np.log10(work["q_value_BH"].clip(lower=1e-300))
    fig2 = px.scatter(
        work, x="prevalence", y="epsilon_squared", size="-log10(q)", color="study_with_highest_median_CLR",
        hover_name="feature", color_discrete_map=STUDY_COLOR_MAP, size_max=32,
        title="Prevalence vs. effect size",
        labels={"prevalence": "Prevalence (share of samples)", "epsilon_squared": "Effect size (ε²)",
                "study_with_highest_median_CLR": "Highest in"},
    )
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(style_fig(fig2, 500), use_container_width=True)

    section("Genus spotlight", "Choose a genus to see where it peaks.")
    feature = st.selectbox("Genus", work["feature"].tolist())
    row = work[work["feature"] == feature].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prevalence", f"{row['prevalence'] * 100:.1f}%")
    k2.metric("Effect size (ε²)", f"{row['epsilon_squared']:.3f}")
    k3.metric("q-value", f"{row['q_value_BH']:.1e}")
    k4.metric("Peaks in", str(row["study_with_highest_median_CLR"]))

    med = pd.DataFrame({
        "Study": [c.replace("median_CLR_", "") for c in median_cols],
        "Median CLR": [row[c] for c in median_cols],
    }).sort_values("Median CLR", ascending=False)
    fig3 = px.bar(med, x="Study", y="Median CLR", color="Study", color_discrete_map=STUDY_COLOR_MAP,
                  title=f"{feature}: median CLR across studies")
    fig3.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(style_fig(fig3, 380), use_container_width=True)

    if not robust.empty:
        stable = robust[robust["significant_all_three"] & robust["highest_CLR_study_stable_all_three"]]
        section("Consistent signatures",
                f"{len(stable)} genera keep both their significance and their top study under every "
                f"zero-replacement setting.")
        show = stable[["feature", "prevalence", "highest_CLR_study_1e-5", "median_q_value"]].rename(columns={
            "feature": "Genus", "prevalence": "Prevalence",
            "highest_CLR_study_1e-5": "Highest in", "median_q_value": "Median q-value",
        })
        show["Prevalence"] = show["Prevalence"] * 100
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            column_config={
                "Prevalence": st.column_config.ProgressColumn("Prevalence", format="%.1f%%", min_value=0, max_value=100),
                "Median q-value": st.column_config.NumberColumn(format="%.1e"),
            },
        )


def page_map() -> None:
    hero(
        "Community map",
        "A bird's-eye view of how similar milk microbiomes are to one another.",
        ["Bray–Curtis", "PCoA"],
    )
    explain(
        "<b>How to read this map.</b> Every dot is a milk sample. Samples that sit <b>close together</b> have similar "
        "microbial communities; samples far apart are different. Colours mark the study each sample came from. "
        "Distinct clusters show that each cohort has its own characteristic community structure."
    )
    left, right = st.columns([3, 2])
    with left:
        show_image("mami_bray_curtis_pcoa.png", "Bray–Curtis principal coordinates analysis of MAMI milk samples")
    with right:
        st.markdown(card("🧭", "What the clusters tell us",
                         "Each study forms a recognisable region of the map, which is why a model can learn to "
                         "tell them apart. This is the visual counterpart of the Model insights page."),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(card("📏", "About the axes",
                         "PCoA compresses hundreds of genus measurements into two dimensions. The axes have no "
                         "units; only the distance between dots matters."),
                    unsafe_allow_html=True)


def page_model() -> None:
    hero(
        "Model insights",
        "Can a machine-learning model recognise where a milk sample came from using microbes alone?",
        ["Cross-validated", "Logistic regression", "Random forest"],
    )
    if ml_metrics.empty:
        st.info("Add the classifier metrics file to the data folder to unlock this page.")
        return

    explain(
        "<b>What was tested.</b> Models were trained to predict the originating <b>study</b> of each sample from its "
        "genus-level abundances, then evaluated with <b>five-fold cross-validation</b> (always on samples the model "
        "had not seen). <b>Balanced accuracy</b> gives every study equal weight, so smaller cohorts count as much as "
        "large ones. Two naive baselines are shown for comparison."
    )

    best = ml_metrics.sort_values("balanced_accuracy", ascending=False).iloc[0]
    baseline = ml_metrics[ml_metrics["model"].str.contains("Dummy")]["balanced_accuracy"].max()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", best["model"].replace("Regularized ", ""))
    k2.metric("Accuracy", f"{best['accuracy'] * 100:.1f}%")
    k3.metric("Balanced accuracy", f"{best['balanced_accuracy'] * 100:.1f}%")
    k4.metric("Lift over baseline", f"{best['balanced_accuracy'] / baseline:.1f}×")

    long = ml_metrics.melt(id_vars="model", value_vars=["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"],
                           var_name="Metric", value_name="Score")
    long["Metric"] = long["Metric"].map({"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                                         "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"})
    fig = px.bar(long, x="model", y="Score", color="Metric", barmode="group", title="Model comparison",
                 color_discrete_sequence=[TEAL, CORAL, "#2563EB", "#9333EA"])
    fig.update_layout(xaxis_title="", yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    if not class_metrics.empty:
        section("Performance for each study", "How well every cohort is recognised.")
        cm = class_metrics.sort_values("f1_score", ascending=True)
        fig2 = go.Figure()
        for col, name, colr in [("precision", "Precision", TEAL), ("recall", "Recall", CORAL), ("f1_score", "F1", NAVY)]:
            fig2.add_trace(go.Bar(y=cm["study"], x=cm[col], name=name, orientation="h", marker_color=colr))
        fig2.update_layout(barmode="group", title="Precision, recall and F1 per study", xaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig2, 480), use_container_width=True)

    section("Confusion matrix",
            "Rows are the true study, columns the predicted study; the bright diagonal marks correct calls.")
    show_image("mami_study_classifier_confusion_matrix.png")


def page_data() -> None:
    hero(
        "Data & provenance",
        "Where every number on this dashboard comes from.",
        ["MAMI", "PRIME", "Reproducible"],
    )

    section("Source datasets")
    left, right = st.columns(2)
    with left:
        st.markdown(card("🥛", "MAMI: microbiome profiles",
                         "1,128 breast-milk samples with a genus-level abundance matrix and accession metadata "
                         "(study, country, postpartum sampling days, delivery mode where recorded). "
                         "Powers every analysis on this dashboard."),
                    unsafe_allow_html=True)
        if not alpha_study.empty:
            st.markdown("")
            st.caption("Samples per MAMI study")
            fig = px.pie(alpha_study, names="Study", values="n", hole=0.55, color="Study",
                         color_discrete_map=STUDY_COLOR_MAP)
            fig.update_traces(textinfo="label+percent", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with right:
        st.markdown(card("🔬", "PRIME: participant metadata",
                         "176 breast-milk records from two projects (USA and France), each with participant and "
                         "BioSample identifiers and a healthy-donor label. Provides complementary context for "
                         "the human-milk resource."),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(card("🧪", "Methods at a glance",
                         "• Alpha diversity: richness, Shannon, Simpson, Pielou<br>"
                         "• Beta diversity: Bray–Curtis distance with PCoA<br>"
                         "• Feature testing: CLR transform, Kruskal–Wallis, Benjamini–Hochberg correction<br>"
                         "• Machine learning: regularised logistic regression and random forest, 5-fold CV"),
                    unsafe_allow_html=True)

    if not inventory.empty:
        section("Dataset overview")
        cols = [c for c in ["source", "scope", "records_in_current_files", "geographic_coverage",
                            "microbiome_data"] if c in inventory.columns]
        st.dataframe(
            inventory[cols].rename(columns={
                "source": "Source", "scope": "Scope", "records_in_current_files": "Records",
                "geographic_coverage": "Geography", "microbiome_data": "Microbiome data"}),
            use_container_width=True, hide_index=True,
        )

    section("Explore your own table", "Preview any CSV in this session. Nothing is stored.")
    upload = st.file_uploader("Upload a de-identified CSV", type=["csv"])
    if upload is not None:
        try:
            preview = pd.read_csv(upload)
            st.success(f"Loaded {len(preview):,} rows × {len(preview.columns):,} columns.")
            st.dataframe(preview.head(25), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")


EXAMPLE_QUESTIONS = [
    "How does geography shape the human milk microbiome?",
    "How do intrapartum antibiotics and breastfeeding exclusivity affect milk microbiota?",
    "Why are negative controls essential in low-biomass milk sequencing?",
    "What did our MAMI analysis find about diversity across studies?",
]


def page_assistant() -> None:
    hero(
        "Research assistant",
        "Ask a question and get an answer written by Google Gemini, grounded in a curated library of "
        "milk-microbiome papers and our own MAMI/PRIME results, with every claim tied to a source.",
        ["Google Gemini", "Semantic search", "Cited answers"],
    )

    explain(
        "<b>How it works.</b> 1) Your question is converted into a Gemini <b>embedding</b>. "
        "2) The most similar passages in the literature library are retrieved. "
        "3) <b>Gemini</b> writes an answer using only those passages and cites them as [D01], [P01] and so on. "
        "Open <i>Retrieved passages</i> under any answer to check the evidence yourself."
    )

    corpus = load_corpus()
    store_dir = Path(__file__).parent / "rag_store"
    manifest_path = store_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    n_lit = sum(1 for d in corpus if d.get("doc_type") != "internal_project_note")
    n_int = len(corpus) - n_lit
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Papers in library", n_lit)
    m2.metric("Project result notes", n_int)
    m3.metric("Searchable passages", manifest.get("chunk_count", "–"))
    m4.metric("Search index", "Ready ✅" if manifest else "Needs build")

    with st.expander("⚙️ Gemini connection and settings", expanded=not manifest):
        api_key = st.text_input(
            "Gemini API key",
            value=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey. It is used only in memory for this session.",
        )
        s1, s2, s3 = st.columns(3)
        embed_model = s1.selectbox(
            "Embedding model",
            EMBEDDING_MODEL_CHOICES,
            index=EMBEDDING_MODEL_CHOICES.index(DEFAULT_EMBEDDING_MODEL)
            if DEFAULT_EMBEDDING_MODEL in EMBEDDING_MODEL_CHOICES else 0,
        )
        chat_model = s2.selectbox(
            "Answer model",
            CHAT_MODEL_CHOICES,
            index=CHAT_MODEL_CHOICES.index(DEFAULT_CHAT_MODEL) if DEFAULT_CHAT_MODEL in CHAT_MODEL_CHOICES else 0,
            help="Flash is fast and economical; Pro writes more detailed syntheses.",
        )
        top_k = s3.slider("Passages to retrieve", 2, 8, 5)

        b1, b2 = st.columns([1, 2])
        if b1.button("🔄 Build / refresh search index", type="primary", use_container_width=True):
            if not api_key.strip():
                st.error("Enter your Gemini API key first.")
            else:
                try:
                    with st.spinner("Embedding the literature library with Gemini…"):
                        built = build_index(api_key=api_key, embedding_model=embed_model)
                    st.toast(
                        f"Index ready: {built['document_count']} sources, {built['chunk_count']} passages, "
                        f"{built['vector_dimensions']} dimensions.", icon="✅"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Index build failed: {exc}")
        b2.caption("Run once, and again whenever the library or embedding model changes.")

    section("Ask a question")
    st.caption("Try one of these:")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    picked = None
    for col, q in zip(chip_cols, EXAMPLE_QUESTIONS):
        if col.button(q, use_container_width=True, key=f"chip_{q[:12]}"):
            picked = q

    with st.form("rag_question_form", clear_on_submit=True):
        typed = st.text_area(
            "Your research question",
            placeholder="e.g. What evidence links breastfeeding exclusivity to the milk microbiome?",
            height=90,
        )
        submitted = st.form_submit_button("✨ Get a cited answer", type="primary")

    question = picked or (typed.strip() if submitted else "")
    if question:
        if not api_key.strip():
            st.error("Add your Gemini API key in the settings panel above to ask questions.")
        else:
            try:
                with st.spinner("Searching the library and writing a cited answer with Gemini…"):
                    answer, sources = ask_rag(
                        question=question, api_key=api_key, top_k=top_k,
                        embedding_model=embed_model, chat_model=chat_model,
                    )
                st.session_state.setdefault("rag_history", []).append(
                    {"question": question, "answer": answer, "sources": sources, "model": chat_model}
                )
            except Exception as exc:
                st.error(f"Could not complete the request: {exc}")

    history = st.session_state.get("rag_history", [])
    if history:
        section("Conversation")
        for turn in reversed(history):
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(turn["answer"])
                st.caption(f"Answered by {turn['model']} from {len(turn['sources'])} retrieved passages.")
                with st.expander("📎 Retrieved passages and relevance"):
                    for item in turn["sources"]:
                        internal = item["doc_type"] == "internal_project_note"
                        tag = '<span class="tag internal">project note</span>' if internal else '<span class="tag">published</span>'
                        link = f" · <a href='{item['url']}' target='_blank'>open source ↗</a>" if item.get("url") else ""
                        st.markdown(
                            f'<div class="source"><b>[{item["doc_id"]}] {item["title"]}</b>{tag}'
                            f'<div class="meta">{item.get("year") or "n.d."} · relevance {item["score"]:.3f}{link}</div></div>',
                            unsafe_allow_html=True,
                        )
        if st.button("Clear conversation"):
            st.session_state["rag_history"] = []
            st.rerun()

    section("Literature library", "Every source the assistant can cite.")
    if not source_catalog.empty:
        cols = [c for c in ["doc_id", "title", "authors", "year", "journal", "access_type"] if c in source_catalog.columns]
        st.dataframe(
            source_catalog[cols].rename(columns={
                "doc_id": "ID", "title": "Title", "authors": "Authors", "year": "Year",
                "journal": "Journal", "access_type": "Access"}),
            use_container_width=True, hide_index=True,
        )
    st.caption(
        "The library holds concise, curated summaries of peer-reviewed papers plus one internal project-results note. "
        "For research and education only, not clinical advice."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR NAVIGATION AND ROUTING
# ─────────────────────────────────────────────────────────────────────────────
PAGES = {
    "🏠  Home": page_home,
    "🌍  Diversity": page_diversity,
    "🦠  Microbial signatures": page_signatures,
    "🗺️  Community map": page_map,
    "🤖  Model insights": page_model,
    "💬  Research assistant": page_assistant,
    "📚  Data & provenance": page_data,
}

st.sidebar.markdown("## 🧬 Milk Microbiome Atlas")
st.sidebar.caption("Interactive research dashboard")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Data: MAMI · PRIME · AI: Google Gemini")
st.sidebar.caption("For research use. Findings are exploratory and observational.")

PAGES[choice]()

st.markdown(
    '<div class="footer">Human Milk Microbiome Atlas · research prototype · '
    "exploratory, observational analysis, not for clinical use</div>",
    unsafe_allow_html=True,
)
