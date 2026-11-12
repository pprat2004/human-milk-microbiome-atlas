# 🧬 Human Milk Microbiome Atlas

An interactive dashboard for exploring human-milk microbial communities using the **MAMI** and **PRIME** resources
(1,128 milk samples, 7 studies, 6 countries), with a **Google Gemini** research assistant that answers questions from
a curated literature library with source citations.

## Quick start
1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. *(For the assistant)* get a free key at https://aistudio.google.com/apikey and put it in `.env`
   (copy `.env.example`) or paste it into the assistant page
4. `streamlit run dashboard_app.py` (Windows: double-click `run_dashboard.bat`)
5. Open http://localhost:8501

## Pages
| Page | What you get |
|---|---|
| 🏠 Home | Headline numbers, key findings, data sources |
| 🌍 Diversity | Richness, Shannon, Simpson, evenness by study or country |
| 🦠 Microbial signatures | CLR heatmap, effect-size bubble chart, genus spotlight, consistent signatures |
| 🗺️ Community map | Bray–Curtis PCoA of all samples |
| 🤖 Model insights | Cross-validated study-recognition models, per-study scores, confusion matrix |
| 💬 Research assistant | Gemini-powered, cited answers from the literature library |
| 📚 Data & provenance | Source datasets, methods, session-only CSV preview |

## Research assistant (Google Gemini)
- **Embeddings:** `gemini-embedding-2` (768 dimensions) with local cosine search
- **Answers:** `gemini-3.5-flash` by default; `gemini-3.1-pro-preview` and `gemini-3.1-flash-lite` selectable
- **Grounding:** answers use only retrieved passages, cite source IDs, and keep published papers separate from the
  internal project note
- First use: open the page and click **Build / refresh search index** once

See `RAG_SETUP.md` for details, model overrides and command-line tools.

## Methods
Alpha diversity (richness, Shannon, Simpson, Pielou) · Bray–Curtis PCoA · CLR transform with Kruskal–Wallis and
Benjamini–Hochberg correction · regularised logistic regression and random forest with 5-fold cross-validation.

## Files
`dashboard_app.py` (app) · `rag_engine.py` (Gemini RAG) · `data/` (analysis outputs and literature corpus) ·
`rag/` (evaluation questions, corpus provenance) · `ORIGINAL_DATA_TRACK_NOTE.md` (scope notes)

*Research prototype: exploratory and observational, not for clinical use. Keep your API key out of source control.*
