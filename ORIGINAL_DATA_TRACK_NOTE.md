# Original project track: MAMI + PRIME

## Scope decision
The project will continue on the original dataset track using the already supplied MAMI and PRIME files. NCBI BioProject PRJNA1292286 / IMiC is intentionally excluded from the current workstream. The objectives can be revised later after the team reviews the evidence from the original track.

## Current datasets
### MAMI
- 1,128 breast-milk samples.
- Matching abundance matrix and accession metadata.
- Six countries and seven studies.
- Supports exploratory composition, diversity and study/country description.
- Meaningful postpartum sampling days in 770 records, gestational status in 45, and delivery mode in 101.
- Country and study are confounded.
- The abundance `subject` field is a constant label, not an identifiable participant ID.

### PRIME
- 176 breast-milk metadata rows in the supplied `samples.csv`.
- Two projects / two country labels in this subset.
- All rows are labelled Healthy.
- Participant and BioSample fields are present.
- No abundance matrix was supplied; maternal age, BMI and antibiotic-use fields are missing in the supplied milk subset.

## Current project work
1. Continue descriptive geographic and study-level comparisons using MAMI.
2. Explore associations only with maternal/population variables actually present and sufficiently populated.
3. Treat the study classifier as an exploratory source/batch diagnostic, not a clinical predictor.
4. Develop a study-aware and confounder-aware analysis plan; avoid claiming independent country effects where the design cannot identify them.
5. Build the LLM/RAG research assistant using curated peer-reviewed literature and verified project findings.
6. Keep the dashboard as an integrated deliverable, with data provenance, visual analytics, model diagnostics and RAG interaction.

## Explicit current-data limitations
The supplied MAMI and PRIME files do not contain neonatal feeding behaviour labels, detailed socioeconomic variables, environmental exposure measurements or infant growth outcomes. Therefore, no valid supervised feeding-behaviour or infant-outcome prediction model can be trained from these files. These gaps will be reported transparently. Do not create synthetic labels or infer clinical outcomes from country/study identity.

## Dashboard implementation
The Streamlit dashboard (Human Milk Microbiome Atlas) is structured around:
- Overview
- Geographic & diversity
- Microbial feature explorer
- ML diagnostics
- Original datasets & scope
- Data readiness & uploads

Pages: Home, Diversity, Microbial signatures, Community map, Model insights, Research assistant (Gemini RAG), Data & provenance.
