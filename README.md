# 🌍 Intelligent Excel Data Enricher (AI & Web Search Pipeline)

An advanced, scalable asynchronous Python pipeline designed to enrich Excel datasets (up to 1,000+ rows) by intelligently planning search queries, batching web requests, caching results, and extracting structured data using Large Language Models (via OpenRouter) and Google Search (via Serper.dev).

---

## 🏗️ Architecture & Production Features

To handle large datasets efficiently and prevent API rate limits, token waste, and formatting loss, the system incorporates **6 core architectural pillars**:

1. **Planner + Parallel Executor Pattern:** 
   - A single LLM call analyzes the table headers and task description to generate precise search query templates, fallback queries, and extraction rules.
   - The dataset is processed asynchronously using `asyncio` and `httpx` with a controlled concurrency semaphore (`MAX_CONCURRENT_LLM`).
2. **Serper API Request Batching:** 
   - Instead of single-row requests, web search queries are batched (up to 50 queries per HTTP POST request) to drastically reduce network overhead.
3. **Dual-Layer Local SQLite Caching & Deduplication:** 
   - Uses a local SQLite database (`enrichment_cache.sqlite`) to store raw search snippets and LLM extractions. Identical requests are instantly resolved from the cache, saving time, tokens, and API credits.
4. **Resilient Checkpointing:** 
   - Progress is automatically saved in chunks (`CHECKPOINT_INTERVAL = 50`) directly to the Excel file. If interrupted (network failure or API limits), restarting the script automatically skips completed rows.
5. **Hybrid Verification & Fallback Agent:** 
   - The extraction model evaluates its own confidence (`high`, `medium`, `low`) and verifies units (e.g., converting miles to kilometers or feet to meters). If confidence is low or data is missing, a secondary fallback search query is automatically executed.
6. **Formatting Preservation (`openpyxl`):** 
   - Unlike standard Pandas exports that strip styling, the pipeline preserves original fonts, column widths, and formulas, while visually highlighting low-confidence cells (yellow) and missing data (light red/pink).

---

## 🛠️ Tech Stack

- **Core:** Python 3.11+ (AsyncIO)
- **AI / LLM:** OpenAI Async SDK, OpenRouter (GPT-4o-mini / Gemini Flash)
- **Web Search:** Serper.dev API
- **Data & Excel:** Pandas, Openpyxl
- **Validation:** Pydantic v2
- **Progress Tracking:** TQDM

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.2com/Andre7425/excel-data-enricher.git
cd excel-data-enricher