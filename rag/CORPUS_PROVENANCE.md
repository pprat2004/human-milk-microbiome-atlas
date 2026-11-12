# RAG knowledge base provenance

The RAG starter corpus is stored in `data/literature_corpus.json`.

It contains eight peer-reviewed papers/reviews and one internal project note. The literature entries are concise paraphrased research notes compiled from published abstracts and accessible article material, with DOI/PMCID and source URLs retained. The corpus is not a collection of complete article texts. Before extending it, record the exact version, access type/licence and ingestion date for each source.

The internal note P01 summarizes the already completed MAMI/PRIME data audits and exploratory analyses. It is explicitly tagged `internal_project_note`; the RAG system is instructed not to present it as peer-reviewed literature.

## Current corpus
- D01: Sun et al., 2025 — global geographic human-milk bacteriome/mycobiome variation.
- D02: Stinson et al., 2025 — GUSTO human-milk microbiome, intrapartum antibiotics and breastfeeding exclusivity.
- D03: Spreckels et al., 2023 — low-biomass human-milk extraction and sequencing methods.
- D04: Eisenhofer et al., 2019 — low-biomass microbiome contamination review.
- D05: Salter et al., 2014 — reagent/laboratory contamination and negative controls.
- D06: Dombrowska-Pali et al., 2024 — human-milk microbiome review.
- D07: Naspolini et al., 2025 — exclusive breastfeeding and infant gut microbiome maturation by delivery mode (infant stool microbiome, not milk).
- D08: Patangia et al., 2025 — feeding duration and infant gut microbiome pilot study.
- P01: internal MAMI/PRIME analysis summary; not peer reviewed.

The corpus can be extended with papers from the literature review in the original project presentation. New documents should include stable IDs, full bibliographic metadata, paraphrased evidence notes, study design, sample size, outcomes, limitations, and canonical URL/DOI.
