# Comprehensive Implementation Issues & Technical Debt Catalog

**Project Title:** AI-Powered Agentic System for Political Misinformation Detection, Influence Analysis and Propagation Mitigation  
**Audit Scope:** Codebase, Data Pipelines, Indexing Engines, ML Models, API Security, and Frontend  
**Priority Hierarchy:** Critical (P0) $\to$ High (P1) $\to$ Medium (P2) $\to$ Low (P3)  

---

## 1. Issue Prioritization Matrix

| Issue ID | Severity | Module / Component | Brief Description | Effort Est. | Owner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISSUE-01** | **High (P1)** | ChromaDB Vector Store (`backend/rag_engine.py`) | Partial vector indexing (4,250 of 10,240 LIAR-PLUS training records stored). | 3 Hours | Member 2 |
| **ISSUE-02** | **Medium (P2)** | Machine Learning Evaluation (`notebooks/`, `ml/`) | Missing standalone CLI evaluation script for official LIAR test split. | 2.5 Hours | Member 1 |
| **ISSUE-03** | **Medium (P2)** | Backend Security (`backend/main.py`) | Permissive wildcard CORS (`allow_origins=["*"]`) enabled by default. | 1 Hour | Member 3 |
| **ISSUE-04** | **Medium (P2)** | Influence Analyzer (`backend/influence_analyzer.py`) | Hardcoded maximum year bound (`MAX_VALID_YEAR = 2026`) in Snowflake validation. | 0.5 Hours | Member 2 |
| **ISSUE-05** | **Low (P3)** | Frontend Modularity (`frontend/src/App.jsx`) | Monolithic single-file architecture (>2,200 lines across 12 components). | 3.5 Hours | Member 3 |
| **ISSUE-06** | **Low (P3)** | Upstream Dataset (`dataset/politifact_evidence/`) | Documented upstream LIAR-PLUS TSV row-shift discrepancy in raw scraping. | Managed | Member 1 & 2 |

---

## 2. Detailed Technical Issue Catalog

### ISSUE-01: Partial Vector Indexing in ChromaDB (4,250 of 10,240 Records)
- **Severity:** High (P1)
- **Component:** `backend/rag_engine.py` (Lines 135–236) & `dataset/politifact_evidence/chroma_db/`
- **Root Cause:**
  During Phase 4 development, the initial bulk ingestion of `train_evidence.jsonl` was throttled/halted at 4,250 records to prevent local CPU starvation during test execution. The persistent SQLite database `dataset/politifact_evidence/chroma_db/chroma.sqlite3` currently reports `document_count = 4250`, while the training set contains 10,240 total records.
- **System Impact:**
  Hybrid RAG queries currently search $\approx 41.5\%$ of the total available LIAR-PLUS training evidence corpus. While search operates correctly and returns valid historical citations, the remaining $\approx 58.5\%$ of historical fact-checks are omitted from semantic vector recall.
- **Proposed Technical Solution:**
  1. Develop a dedicated non-blocking batch indexing utility `scripts/build_full_vector_index.py`.
  2. Implement checkpointing and configurable batch size (`batch_size=200`) with brief sleep intervals to prevent CPU throttling.
  3. Resume ingestion from record 4,251 to 10,240, ensuring unique `statement_id` primary keys prevent duplicate insertions.
- **Verification & Testing Steps:**
  1. Run `python scripts/build_full_vector_index.py`.
  2. Call `GET /rag/stats` and verify `"document_count": 10240`.
  3. Execute `pytest tests/test_rag_and_agents.py` to confirm all hybrid retrieval tests pass with full index coverage.

---

### ISSUE-02: Missing Standalone Test Set Evaluation Script
- **Severity:** Medium (P2)
- **Component:** `ml/` (Currently empty) & `notebooks/04_distilbert_training.ipynb`
- **Root Cause:**
  The DistilBERT model and baseline TF-IDF Logistic Regression model were trained, evaluated, and saved via Jupyter Notebooks (`notebooks/03_baseline_model.ipynb` and `notebooks/04_distilbert_training.ipynb`). However, no standalone, reproducible command-line evaluation script exists in the repository.
- **System Impact:**
  During an academic project defense or viva examination, external examiners frequently request live verification of model performance metrics (Accuracy, Macro F1, Precision, Recall, Confusion Matrix) on the official test set (`dataset/liar/test.tsv`). Relying on opening and re-running a large Jupyter notebook during a presentation creates operational friction.
- **Proposed Technical Solution:**
  Create a clean, dedicated CLI evaluation script `scripts/evaluate_models.py` that:
  1. Loads `dataset/liar/test.tsv` (1,267 test records).
  2. Runs batch inference through `baseline_model.pkl` and `distilbert_liar_final`.
  3. Computes and prints formatted classification reports (`sklearn.metrics.classification_report`) with macro and weighted averages.
  4. Saves an exportable confusion matrix image and JSON report to `reports/model_evaluation_report.json`.
- **Verification & Testing Steps:**
  1. Run `python scripts/evaluate_models.py --model all --split test`.
  2. Verify that output metrics match reported benchmark numbers (DistilBERT Accuracy $\approx 27-29\%$ on 6-class fine-grained LIAR, Macro F1 $\approx 0.26-0.28$, baseline $\approx 24-25\%$).

---

### ISSUE-03: Permissive Wildcard CORS Configuration
- **Severity:** Medium (P2)
- **Component:** `backend/main.py` (Lines 74–80)
- **Root Cause:**
  ```python
  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
  The wildcard origin `["*"]` was introduced during initial scaffolding to avoid local origin mismatch errors between the Vite dev server (`http://localhost:5173`) and the FastAPI backend.
- **System Impact:**
  In a production or deployed environment, wildcard CORS allows arbitrary third-party web domains to issue credentialed AJAX requests against the inference API, violating standard web security best practices.
- **Proposed Technical Solution:**
  Refactor CORS origin resolution to read from an environment variable with a safe localhost fallback:
  ```python
  import os
  allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000")
  allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
  app.add_middleware(
      CORSMiddleware,
      allow_origins=allowed_origins,
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )
  ```
- **Verification & Testing Steps:**
  1. Update CORS origins.
  2. Start FastAPI and verify frontend UI communicates seamlessly at `http://localhost:5173`.
  3. Execute `pytest tests/test_frontend_backend_integration.py` to confirm integration tests pass.

---

### ISSUE-04: Static Hardcoded Year Bound in Twitter Snowflake Decoding
- **Severity:** Medium (P2)
- **Component:** `backend/influence_analyzer.py` (Line 28)
- **Root Cause:**
  ```python
  MAX_VALID_YEAR = 2026  # Bound for valid decodable timestamps
  ```
  The upper year limit for validating decoded Snowflake timestamps is hardcoded to 2026.
- **System Impact:**
  If the application is inspected or evaluated in calendar year 2027 or beyond, valid newly generated Twitter/X Snowflake IDs would be rejected by `decode_snowflake_id()` and classified as `legacy` or invalid.
- **Proposed Technical Solution:**
  Dynamically set `MAX_VALID_YEAR` based on current system time with a 1-year forward tolerance:
  ```python
  MAX_VALID_YEAR = datetime.datetime.now(datetime.timezone.utc).year + 1
  ```
- **Verification & Testing Steps:**
  1. Replace hardcoded constant with dynamic expression.
  2. Run `pytest tests/test_influence_analysis.py`.
  3. Verify that test cases validating ID `1178658045934415872` (Sept 2019) continue passing.

---

### ISSUE-05: Monolithic Single-File Frontend Architecture
- **Severity:** Low (P3)
- **Component:** `frontend/src/App.jsx` (Lines 1–2207)
- **Root Cause:**
  All 12 React components (`PrimaryAssessmentSection`, `AgentExecutionTraceSection`, `VerificationRiskSection`, `MitigationRecommendationsSection`, `EvidenceCard`, `HistoricalEvidenceSection`, `InfluenceCard`, `HistoricalInfluenceSection`, `Phase1ComparisonSection`, etc.) were consolidated into `App.jsx` during rapid prototyping to maintain single-file state cohesion.
- **System Impact:**
  While the application compiles cleanly without errors and Vite builds in 6.8 seconds, maintaining a 2,200+ line JSX file creates merge conflicts when multiple developers edit the frontend concurrently.
- **Proposed Technical Solution:**
  Refactor the component tree into dedicated files under `frontend/src/components/`:
  - `components/PrimaryAssessment.jsx`
  - `components/AgentExecutionTrace.jsx`
  - `components/VerificationRisk.jsx`
  - `components/MitigationRecommendations.jsx`
  - `components/HistoricalEvidence.jsx`
  - `components/HistoricalInfluence.jsx`
  - `components/Phase1Comparison.jsx`
  Leave `App.jsx` as the clean root coordinator (<150 lines).
- **Verification & Testing Steps:**
  1. Decompose components into `src/components/`.
  2. Run `npx oxlint` to ensure 0 lint errors.
  3. Run `npm run build` to confirm clean Vite compilation.

---

### ISSUE-06: Upstream LIAR-PLUS TSV Row-Shift Flaw in Source Data
- **Severity:** Low (P3) / Academic Phenomenon
- **Component:** `dataset/politifact_evidence/raw/` and `backend/evidence_retriever.py`
- **Root Cause:**
  The official LIAR-PLUS dataset published in 2018 (Alhindi et al.) suffers from documented scraping anomalies: embedded tabs and newlines in PolitiFact justification text caused occasional column-shifting during TSV formatting. Consequently, in $\approx 10-14\%$ of raw records, the text in the justification column belongs to a different row's fact-check.
- **System Impact:**
  If justifications are retrieved naively, the system would present justification text that does not match the historical claim or speaker.
- **Current Handling & Academic Strength:**
  This issue has been handled by our custom heuristic:
  ```python
  check_claim_justification_alignment(claim, speaker, justification)
  ```
  This function programmatically verifies whether the speaker's name or informative claim keywords exist inside the justification text. Misaligned records are flagged with `has_context_alignment: false` and surfaced with a warning badge in the UI.
- **Viva Defense Guidance:**
  This is NOT a bug in our code, but rather a known published defect in the benchmark dataset. Detecting and surfacing this anomaly is an academic strength that demonstrates rigor during project evaluation.

---

## 3. Recommended Implementation Roadmap

```
Week 1:
  ├─ Member 2: Run build_full_vector_index.py to reach 10,240 ChromaDB records (ISSUE-01).
  ├─ Member 1: Write scripts/evaluate_models.py for CLI test split reports (ISSUE-02).
  └─ Member 3: Patch CORS origin resolution and dynamic MAX_VALID_YEAR (ISSUE-03, ISSUE-04).

Week 2:
  ├─ Member 3: Optional component modularization of App.jsx into src/components/ (ISSUE-05).
  └─ Team: Rehearse viva defense on upstream TSV row-shift detection (ISSUE-06).
```

