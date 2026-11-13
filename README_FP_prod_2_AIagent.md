# 🌍 Intelligent Excel Data Enricher (AI & Web Search Pipeline)

An advanced, production-grade asynchronous Python pipeline designed to enrich Excel datasets (up to 1,000+ rows) by intelligently planning search queries, batching web requests, caching results, and extracting structured data using Large Language Models (via OpenRouter) and Google Search (via Serper.dev).

The system features a **dual interface** (Command Line Interface & FastAPI REST API), a built-in **Cost Tracker** for precise API budget monitoring, and resilient checkpointing.

---

## 🏗️ Architecture & Core Features

1. **Planner + Parallel Executor Pattern:** 
   - A single LLM call analyzes the dataset headers and the user's natural language task description, automatically generating search templates, fallback queries, and extraction instructions.
2. **Serper API Batching & SQLite Caching:** 
   - Web search queries are batched (up to 50 queries per HTTP POST request). A local SQLite cache (`enrichment_cache.sqlite`) ensures that identical queries are never charged twice, saving tokens and API credits.
3. **Cost Tracker & Token Counter:** 
   - Automatically tracks LLM prompt/completion tokens and Serper search queries across threads, calculating exact expenses in USD based on current pricing models (e.g., GPT-4o-mini).
4. **Resilient Checkpointing:** 
   - Progress is saved in chunks directly to the Excel file. If interrupted, restarting the script automatically skips completed rows.
5. **Formatting Preservation (`openpyxl`):** 
   - Preserves original fonts, column widths, and formulas while visually highlighting low-confidence cells (yellow) and missing data (light pink).
6. **Dual Interface (CLI & FastAPI):** 
   - Run tasks directly from the terminal or deploy as a microservice with an interactive Swagger UI (`/docs`).

---

## 🛠️ Tech Stack

- **Core:** Python 3.11+ / 3.14 (Requests Session / ThreadPoolExecutor)
- **AI / LLM:** OpenRouter API (`openai/gpt-4o-mini`)
- **Web Search:** Serper.dev API
- **Web Framework:** FastAPI & Uvicorn
- **Data & Excel:** Pandas, Openpyxl
- **Validation:** Pydantic v2

---

## 🚀 Getting Started

### 1. Clone & Set Up Virtual Environment
```bash
git clone [https://github.com/Andre7425/excel-data-enricher.git](https://github.com/Andre7425/excel-data-enricher.git)
cd excel-data-enricher

python3 -m venv .venv
source .venv/bin/activate