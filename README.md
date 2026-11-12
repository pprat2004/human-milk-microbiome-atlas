# 🧬 Human Milk Microbiome Atlas

**A study-aware, reproducible analysis of human-milk microbiome diversity, composition and population variation, with an interactive dashboard and a Gemini-powered, source-cited research assistant.**

![Python](https://img.shields.io/badge/Python-3.10%2B-0F766E)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-F97316)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-2563EB)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This project analyses **1,128 human-milk microbiome samples from 7 studies in 6 countries** (MAMI database), with
participant metadata from **PRIME**. It combines classical microbiome statistics, machine learning and
retrieval-augmented generation (RAG) in one dashboard.

> **Title:** AI-Driven Analysis of Human Milk Microbiome Diversity, Population-Level Variation, Maternal Factors, and
> Clinical Associations Using Machine Learning and Retrieval-Augmented Generation

## Key contributions
- **Audit-first, study-aware design:** dataset feasibility is checked before any modelling, and results are interpreted
  at the study level.
- **Integrated pipeline:** diversity, CLR composition, Bray–Curtis PCoA, cross-validated ML and a RAG assistant in a
  single reproducible app.
- **Study signatures are strong:** a classifier recovers the originating study with ~95% accuracy (balanced accuracy
  ~91%, versus ~14% for a naive baseline).
- **Study outweighs country:** study explains more diversity variance than country (Shannon η² 0.153 vs 0.065), and the
  two US studies differ sharply from each other.
- **Consistent signatures:** 49 genera keep their significance and top study across all zero-replacement settings.
- **Cited AI assistant:** Gemini answers questions from a curated literature library and keeps published papers separate
  from this project's own results.

## Dashboard pages
| Page | Description |
|---|---|
| 🏠 Home | Headline numbers and findings |
| 🌍 Diversity | Richness, Shannon, Simpson and evenness by study or country |
| 🦠 Microbial signatures | CLR heatmap, effect sizes, genus spotlight |
| 🗺️ Community map | Bray–Curtis PCoA of all samples |
| 🤖 Model insights | Cross-validated study-recognition models |
| 💬 Research assistant | Cited answers from Google Gemini (RAG) |
| 📚 Data & provenance | Sources, methods, CSV preview |

## Quick start
```bash
git clone https://github.com/<your-username>/human-milk-microbiome-atlas.git
cd human-milk-microbiome-atlas
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your GEMINI_API_KEY
streamlit run dashboard_app.py
```
Open http://localhost:8501. For the assistant, get a free key at https://aistudio.google.com/apikey, open
**💬 Research assistant** and click **Build / refresh search index** once.

## Methods
| Step | Approach |
|---|---|
| Alpha diversity | Observed richness, Shannon, Simpson, Pielou evenness |
| Beta diversity | Bray–Curtis distance, PCoA |
| Differential composition | CLR transform, Kruskal–Wallis, Benjamini–Hochberg FDR, ε² effect size |
| Robustness | Three zero-replacement settings (1e-6, 1e-5, 1e-4) |
| Machine learning | Regularised logistic regression, random forest, 5-fold CV, naive baselines |
| RAG | `gemini-embedding-2` (768-d) + local cosine search + `gemini-3.5-flash` grounded generation |

## Repository structure
```
├── dashboard_app.py          # Streamlit dashboard
├── rag_engine.py             # Gemini RAG: chunking, embeddings, retrieval, generation
├── build_rag_index.py        # Build the vector index from the CLI
├── evaluate_rag_retrieval.py # Hit@3 / MRR@5 retrieval evaluation
├── test_rag_corpus.py        # Corpus validation (no API call)
├── data/                     # Analysis outputs, figures, literature corpus
├── rag/                      # Evaluation questions, corpus provenance
├── requirements.txt
└── .env.example
```

## Interpretation notes
- In MAMI, country and study are largely the same variable (five of six countries come from a single study), so
  **country-specific effects cannot be separated from study-level effects**. Differences are reported per study.
- Study recognition by ML is a **batch-effect diagnostic**, not a clinical predictor.
- This is an exploratory, observational research prototype and **not for clinical use**.

## Data sources
- **MAMI** – human-milk microbiome database (bioinfo.ioz.ac.cn/mami)
- **PRIME** – breast-milk metadata resource (primedb.sjtu.edu.cn)

Please check each database's terms of use before redistributing raw data. This repository contains derived analysis
outputs and concise literature summaries, not full-text papers.

## Roadmap
- Harmonised maternal and clinical metadata and leave-one-study-out validation
- PRIME abundance matrix integration
- Expanded literature corpus and RAG evaluation set
- Docker deployment and a FastAPI backend

## Citation
See `CITATION.cff`. 

## License
MIT. See `LICENSE`.
