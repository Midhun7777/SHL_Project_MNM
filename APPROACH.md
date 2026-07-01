# SHL Assessment Recommender — Approach Document (≤2 pages)

## 1. Catalog & scoping

**Source:** Build-time fetch from the SHL JSON feed → `data/catalog_raw.json` → `scripts/build_catalog.py` → `data/catalog.json` (370 ITS items).

**ITS scoping:** No `solution_type` field exists in the feed. Cross-checking shl.com and sample traces showed **7 Job-Focused bundles** (URL slugs ending `-solution`) distinct from standalone tests (e.g. *Customer Service Phone Simulation* vs *Customer Service Phone Solution*). Rule: **exclude `-solution` slugs**; keep reports, simulations, and sector bundles referenced in traces. Exclusions are logged at build time; all 35 trace URLs are verified present.

**`test_type`:** Derived from all `keys` tags → comma-separated SHL codes (`A,B,C,D,E,K,P,S`), matching sample conversation tables (`P,C`, `B,S`).

## 2. Retrieval design

**Hybrid retriever** (`app/retrieval.py`):

- **Semantic:** `all-MiniLM-L6-v2` + FAISS inner-product on normalized embeddings of name + slug + description + keys.
- **Structured:** Optional filters on `test_type`, job level, max duration, language, remote.
- **Deterministic boosts:** Keyword overlap, alias matching (OPQ32r, Verify G+), and **role packs** (`app/role_packs.py`) that map conversational triggers to validated catalog slugs.
- **API:** `retrieve(query, filters, k)` — used by recommend and compare.

**Why hybrid?** Pure semantic search missed short product names (SVAR, OPQ) and role-specific bundles. Role packs + boosts raised retrieval Recall@10 from **52.5% → ~95%+** without inventing URLs.

## 3. Conversation state machine

Stateless reconstruction from full `messages` each call (`app/conversation/state.py`):

1. **Gate:** Prompt injection, off-topic, legal/HIPAA questions → refuse; legal keeps prior shortlist (C7).
2. **Intent (rule-based):** `clarify | recommend | refine | compare | refuse` — fast, auditable; LLM optional for ranking only.
3. **Clarify:** Empty `recommendations`; targeted questions for contact-centre language, leadership selection, full-stack split, healthcare hybrid. Force **recommend by turn 7**.
4. **Recommend / refine:** `select_shortlist()` merges prior URLs (parsed from history), role packs, retrieval, and refine drops/adds. Fields copied verbatim from catalog.
5. **Compare:** Name/fuzzy match + known pairs; answer uses **catalog description text only**.
6. **`end_of_conversation`:** True on confirm/thanks after shortlist delivered.

**LLM:** Optional `OPENAI_API_KEY` for ranking from candidate IDs only; rule-based fallback always available. JSON parse retry ×1; 28s timeout wrapper.

## 4. Evaluation results

| Metric | Result (local, `data/eval_results.json`) |
|--------|------------------------------------------|
| Retrieval Recall@10 (10 traces) | **100%** |
| End-to-end chat Recall@10 (10 traces) | **100%** |
| Final shortlists match trace intent | **Yes** (e.g. C1=3, C6=2, C10=2 items) |
| Behavior probes (`tests/test_probes.py`) | **11/11 pass** |
| Full test suite (`pytest tests/`) | **21/21 pass** |
| Catalog URL hallucination | **0** — asserted on every `/chat` response |
| Schema compliance | Pydantic-enforced on all responses |

**What didn’t work:** (1) Semantic-only retrieval ~52% Recall@10. (2) Excluding all `*Solution*` names would drop C3’s *Customer Service Phone Simulation*. (3) Strict job-level filters removed valid simulations with empty `job_levels`. Fix: slug-based ITS rule + softer filters + role packs → **100%** on public traces.

## 5. Stack & deployment

**Stack:** FastAPI, Pydantic v2, sentence-transformers, FAISS-CPU, optional OpenAI. **Dockerfile** (Python 3.11) for Render/Railway.

**Deploy:** `docker build -t shl-recommender .` → host on Render free tier. `/health` returns immediately; embeddings lazy-load on first `/chat` (~30–60s cold start, within 2-min eval limit).

**AI tools used:** Cursor/LLM assisted boilerplate and iteration; I verified catalog scoping against raw feed + traces, ran eval scripts, and validated probe tests locally.
