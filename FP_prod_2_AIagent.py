import os
import re
import json
import time
import sqlite3
import shutil
import argparse
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from pydantic import BaseModel, Field
from tqdm import tqdm
from dotenv import load_dotenv

# FastAPI імпорти
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

# ==========================================
# 1. КОНФІГУРАЦІЯ ТА БЕЗПЕЧНЕ ЗАВАНТАЖЕННЯ КЛЮЧІВ
# ==========================================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ Помилка: OPENROUTER_API_KEY не знайдено у файлі .env")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    raise ValueError("❌ Помилка: SERPER_API_KEY не знайдено у файлі .env")

OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip().strip('"').strip("'")
SERPER_API_KEY = SERPER_API_KEY.strip().strip('"').strip("'")

# Записуємо URL через конкатенацію, щоб при копіюванні не додавалися Markdown-дужки [url](url)
OPENROUTER_URL = "https://" + "openrouter.ai/api/v1/chat/completions"
SERPER_URL = "https://" + "google.serper.dev/search"

MODEL_NAME = "openai/gpt-4o-mini"
CACHE_DB_PATH = "enrichment_cache.sqlite"
SERPER_BATCH_SIZE = 50
CHECKPOINT_INTERVAL = 50
MAX_WORKERS = 10

# Тарифи для розрахунку вартості (USD)
PRICE_PROMPT_PER_1M = 0.15      # $0.15 за 1M вхідних токенів (gpt-4o-mini)
PRICE_COMPLETION_PER_1M = 0.60  # $0.60 за 1M вихідних токенів (gpt-4o-mini)
PRICE_SERPER_PER_1K = 1.00      # $1.00 за 1000 пошукових запитів Serper

# Кольори для маркування комірок в Excel
FILL_LOW_CONFIDENCE = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # Жовтий
FILL_MISSING = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")         # Рожевий


def sanitize_url(url: str) -> str:
    """Очищує URL, якщо буфер обміну випадково перетворив його на '[https://...](https://...)'."""
    url = url.strip()
    md_match = re.match(r"^\[.*?\]\((https?://[^\)]+)\)$", url)
    if md_match:
        return md_match.group(1)
    return url.strip("[]()")


# ==========================================
# 2. ПОТОКОБЕЗПЕЧНИЙ ТРЕКЕР ВАРТОСТІ (COST TRACKER)
# ==========================================
@dataclass
class CostTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    llm_cache_hits: int = 0
    serper_queries: int = 0
    serper_cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_llm_usage(self, usage_dict: Dict[str, Any]):
        with self._lock:
            self.llm_calls += 1
            if usage_dict:
                self.prompt_tokens += usage_dict.get("prompt_tokens", 0) or 0
                self.completion_tokens += usage_dict.get("completion_tokens", 0) or 0

    def add_llm_cache_hit(self):
        with self._lock:
            self.llm_cache_hits += 1

    def add_serper_queries(self, count: int):
        with self._lock:
            self.serper_queries += count

    def add_serper_cache_hit(self):
        with self._lock:
            self.serper_cache_hits += 1

    @property
    def llm_cost_usd(self) -> float:
        return (
            (self.prompt_tokens / 1_000_000) * PRICE_PROMPT_PER_1M +
            (self.completion_tokens / 1_000_000) * PRICE_COMPLETION_PER_1M
        )

    @property
    def serper_cost_usd(self) -> float:
        return (self.serper_queries / 1_000) * PRICE_SERPER_PER_1K

    @property
    def total_cost_usd(self) -> float:
        return self.llm_cost_usd + self.serper_cost_usd

    def print_summary(self):
        print("\n💰 СТАТИСТИКА ВИКОРИСТАННЯ API ТА ВАРТОСТІ:")
        print(f"   ├─ Пошук Serper: {self.serper_queries} запитів (з кешу: {self.serper_cache_hits}) -> ${self.serper_cost_usd:.5f}")
        print(f"   ├─ Запити до LLM: {self.llm_calls} викликів (з кешу: {self.llm_cache_hits})")
        print(f"   ├─ Токени LLM:   {self.prompt_tokens + self.completion_tokens} (In: {self.prompt_tokens}, Out: {self.completion_tokens}) -> ${self.llm_cost_usd:.5f}")
        print(f"   └─ ЗАГАЛЬНА ВАРТІСТЬ: ${self.total_cost_usd:.5f}\n")


# ==========================================
# 3. МОДЕЛІ ДАНИХ (ПІДТРИМКА 1 АБО БАГАТЬОХ КОЛОНОК)
# ==========================================
class ColumnTask(BaseModel):
    target_column: str = Field(description="Точна назва ОДНІЄЇ колонки для запису результату")
    search_query_template: str = Field(description="Основний шаблон запиту англійською з {Назва_Колонки}")
    fallback_query_template: str = Field(description="Альтернативний (ширший) шаблон запиту англійською")
    extraction_instruction: str = Field(description="Сувора інструкція, яке значення та в яких одиницях витягнути")
    expected_type: Literal["number", "string"] = Field(description="Тип очікуваного значення: 'number' або 'string'")


class EnrichmentBatchPlan(BaseModel):
    tasks: List[ColumnTask] = Field(description="Список завдань для кожної окремої колонки")


class ExtractionResult(BaseModel):
    value: Optional[Any] = Field(default=None, description="Витягнуте значення (число або рядок) або null")
    confidence: Literal["high", "medium", "low"] = Field(default="low", description="Рівень впевненості у точності даних")
    unit_verified: bool = Field(default=False, description="Чи відповідає одиниця виміру інструкції")


# ==========================================
# 4. ДОПОМІЖНІ УТИЛІТИ ТА ЛОКАЛЬНИЙ КЕШ SQLITE
# ==========================================
def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_numeric_value(val: Any) -> Optional[float | int]:
    if isinstance(val, (int, float)):
        return val
    if val is None:
        return None
    s = str(val).lower().strip().replace(" ", "")
    multiplier = 1
    if "million" in s or "млн" in s:
        multiplier = 1_000_000
    elif "billion" in s or "млрд" in s:
        multiplier = 1_000_000_000
    elif "thousand" in s or "тис" in s:
        multiplier = 1_000

    if re.match(r"^\d{1,3}(,\d{3})+$", s):
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    match = re.search(r"[-+]?\d*\.?\d+", s)
    if not match:
        return None
    num = float(match.group(0)) * multiplier
    return int(round(num)) if num.is_integer() or multiplier > 1 else round(num, 2)


class LocalCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query TEXT PRIMARY KEY,
                    snippets TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS extraction_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT
                )
            """)

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def get_search(self, query: str) -> Optional[str]:
        norm_q = self.normalize(query)
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT snippets FROM search_cache WHERE query = ?", (norm_q,))
            row = cur.fetchone()
            if row and not row[0].startswith("Search error") and row[0] != "No results found.":
                return row[0]
            return None

    def set_search_bulk(self, items: Dict[str, str]):
        valid_items = [
            (self.normalize(q), s) for q, s in items.items()
            if not s.startswith("Search error") and s != "No results found."
        ]
        if valid_items:
            with self._lock, sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO search_cache (query, snippets) VALUES (?, ?)",
                    valid_items
                )

    def get_extraction(self, query: str, instruction: str) -> Optional[ExtractionResult]:
        key = self.normalize(f"{query}||{instruction}")
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT result_json FROM extraction_cache WHERE cache_key = ?", (key,))
            row = cur.fetchone()
            if row:
                res = ExtractionResult(**json.loads(row[0]))
                if res.value is not None:
                    return res
            return None

    def set_extraction(self, query: str, instruction: str, result: ExtractionResult):
        if result.value is None:
            return
        key = self.normalize(f"{query}||{instruction}")
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO extraction_cache (cache_key, result_json) VALUES (?, ?)",
                (key, result.model_dump_json())
            )


# ==========================================
# 5. НАДІЙНИЙ КЛІЄНТ OPENROUTER ТА SERPER ЧЕРЕЗ REQUESTS
# ==========================================
def call_openrouter_llm(prompt: str, tracker: CostTracker) -> Dict[str, Any]:
    url = sanitize_url(OPENROUTER_URL)
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Excel Data Enricher"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            tracker.add_llm_usage(data.get("usage", {}))
            content = data["choices"][0]["message"]["content"]
            return json.loads(clean_json_text(content))
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Помилка запиту до OpenRouter: {e}") from e
            time.sleep(1.5 ** attempt)
    return {}


def parse_serper_item(data: Dict[str, Any]) -> str:
    snippets = []
    if "answerBox" in data:
        ab = data["answerBox"]
        snippets.append(f"AnswerBox: {ab.get('answer') or ab.get('snippet') or ''}")
    if "knowledgeGraph" in data:
        kg = data["knowledgeGraph"]
        snippets.append(f"KnowledgeGraph: {kg.get('description', '')} | Attrs: {kg.get('attributes', {})}")
    for item in data.get("organic", []):
        snippets.append(f"- {item.get('title', '')}: {item.get('snippet', '')}")
    return "\n".join(snippets) if snippets else "No results found."


def fetch_serper_batches(
    queries: List[str],
    cache: LocalCache,
    tracker: CostTracker
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    missing_queries = []

    seen_normalized = set()
    for q in queries:
        cached = cache.get_search(q)
        if cached is not None:
            results[q] = cached
            tracker.add_serper_cache_hit()
        else:
            norm_q = cache.normalize(q)
            if norm_q not in seen_normalized:
                seen_normalized.add(norm_q)
                missing_queries.append(q)

    if not missing_queries:
        return results

    url = sanitize_url(SERPER_URL)
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    with requests.Session() as session:
        for i in range(0, len(missing_queries), SERPER_BATCH_SIZE):
            batch = missing_queries[i : i + SERPER_BATCH_SIZE]
            payload = [{"q": q, "num": 4} for q in batch]

            for attempt in range(3):
                try:
                    resp = session.post(url, headers=headers, json=payload, timeout=25)
                    resp.raise_for_status()
                    batch_data = resp.json()
                    tracker.add_serper_queries(len(batch))

                    if isinstance(batch_data, dict):
                        batch_data = [batch_data]

                    new_cache_entries = {}
                    for q_str, item_data in zip(batch, batch_data):
                        parsed_text = parse_serper_item(item_data)
                        new_cache_entries[q_str] = parsed_text
                        results[q_str] = parsed_text

                    cache.set_search_bulk(new_cache_entries)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"⚠️ Помилка Serper API: {e}")
                        for q_str in batch:
                            results[q_str] = f"Search error: {str(e)}"
                    time.sleep(1.5 ** attempt)

    for q in queries:
        if q not in results:
            results[q] = cache.get_search(q) or "No results found."

    return results


# ==========================================
# 6. ПЛАНУВАЛЬНИК ТА ГІБРИДНА ВЕРИФІКАЦІЯ
# ==========================================
def create_enrichment_plan(
    df: pd.DataFrame,
    task_description: str,
    tracker: CostTracker
) -> EnrichmentBatchPlan:
    columns = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")

    prompt = f"""
    Проаналізуй структуру Excel-таблиці та завдання користувача.
    Наявні колонки: {columns}
    Приклад перших рядків: {json.dumps(sample_data, ensure_ascii=False)}
    Завдання користувача: "{task_description}"

    ВАЖЛИВО: Якщо користувач просить заповнити одну колонку — створи 1 об'єкт у списку "tasks".
    Якщо користувач просить знайти КІЛЬКА різних показників або заповнити КІЛЬКА колонок (наприклад, і 'population', і 'distance') — створи окремий об'єкт у списку "tasks" для КОЖНОЇ колонки!
    У фігурних дужках шаблонів використовуй ТІЛЬКИ існуючі назви вхідних колонок із {columns}.

    Поверни строго JSON об'єкт із ключем "tasks":
    {{
      "tasks": [
        {{
          "target_column": "назва цільової колонки (напр. 'distance' або 'height')",
          "search_query_template": "шаблон пошуку в Google англійською з {{Назва_Колонки}} (напр. 'straight line distance between {{Capital_From}}, {{Country_From}} and {{Capital_To}}, {{Country_To}} in km')",
          "fallback_query_template": "альтернативний переформульований шаблон англійською (напр. 'air distance {{Capital_From}} to {{Capital_To}} kilometers')",
          "extraction_instruction": "детальна вимога, яке значення повернути (напр. 'Поверни пряму відстань у кілометрах як чисте число без тексту')",
          "expected_type": "number" або "string"
        }}
      ]
    }}
    """

    plan_dict = call_openrouter_llm(prompt, tracker)
    if "tasks" not in plan_dict and "target_column" in plan_dict:
        plan_dict = {"tasks": [plan_dict]}
    return EnrichmentBatchPlan(**plan_dict)


def extract_with_verification(
    query: str,
    snippets: str,
    task_plan: ColumnTask,
    cache: LocalCache,
    tracker: CostTracker
) -> ExtractionResult:
    cached = cache.get_extraction(query, task_plan.extraction_instruction)
    if cached is not None:
        tracker.add_llm_cache_hit()
        return cached

    if not snippets or snippets.startswith("Search error") or snippets == "No results found.":
        return ExtractionResult(value=None, confidence="low", unit_verified=False)

    prompt = f"""
    Пошуковий запит: {query}
    Сніпети з пошуку:
    {snippets}

    Інструкція: {task_plan.extraction_instruction}
    Очікуваний тип: {task_plan.expected_type}

    Проаналізуй сніпети.
    - Якщо очікуваний тип "number", ОБОВ'ЯЗКОВО поверни в полі "value" чисте число (наприклад: 2950000 або 2024, а НЕ "2.95 million" чи "2024 km"). Якщо в тексті милі або фути — переведи математично в км або метри.
    - Якщо цифра є у сніпетах — обов'язково витягни її.

    Поверни строго JSON:
    {{
        "value": <число або рядок без одиниць виміру, або null якщо даних немає>,
        "confidence": "high" | "medium" | "low",
        "unit_verified": true | false
    }}
    """

    try:
        data = call_openrouter_llm(prompt, tracker)
        result = ExtractionResult(**data)

        if task_plan.expected_type == "number" and result.value is not None:
            parsed_num = parse_numeric_value(result.value)
            if parsed_num is not None:
                result.value = parsed_num
            else:
                result.confidence = "low"

        cache.set_extraction(query, task_plan.extraction_instruction, result)
        return result
    except Exception as e:
        print(f"⚠️ Помилка LLM для запиту '{query}': {e}")
        return ExtractionResult(value=None, confidence="low", unit_verified=False)


# ==========================================
# 7. СТВОРЕННЯ *_enriched.xlsx ТА ЗАПИС КОЛОНОК
# ==========================================
def ensure_column_in_workbook(output_path: str, target_column: str) -> int:
    wb = load_workbook(output_path)
    ws = wb.active

    target_col_idx = None
    for col_idx in range(1, ws.max_column + 1):
        cell_val = ws.cell(row=1, column=col_idx).value
        if cell_val and str(cell_val).strip().lower() == target_column.strip().lower():
            target_col_idx = col_idx
            break

    if target_col_idx is None:
        target_col_idx = ws.max_column + 1
        ws.cell(row=1, column=target_col_idx, value=target_column)
        wb.save(output_path)

    wb.close()
    return target_col_idx


def save_checkpoint_to_excel(
    output_path: str,
    target_col_idx: int,
    updates: Dict[int, ExtractionResult]
):
    if not updates:
        return

    wb = load_workbook(output_path)
    ws = wb.active

    for df_idx, res in updates.items():
        excel_row = df_idx + 2
        cell = ws.cell(row=excel_row, column=target_col_idx)
        cell.value = res.value

        if res.value is None:
            cell.fill = FILL_MISSING
        elif res.confidence == "low" or not res.unit_verified:
            cell.fill = FILL_LOW_CONFIDENCE

    wb.save(output_path)
    wb.close()


# ==========================================
# 8. ГОЛОВНИЙ ПАЙПЛАЙН (THREADPOOLEXECUTOR)
# ==========================================
def format_query(template: str, row_dict: Dict[str, Any], instruction: str) -> str:
    clean_dict = {k: ("" if pd.isna(v) else v) for k, v in row_dict.items()}
    try:
        return template.format(**clean_dict).strip()
    except KeyError:
        fallback_str = " ".join([f"{k}: {v}" for k, v in clean_dict.items() if v != ""])
        return f"{fallback_str} {instruction}".strip()


def process_excel_with_stats(
    file_path: str,
    task_description: str,
    resume_checkpoint: bool = False
) -> tuple[str, CostTracker]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Початковий файл '{file_path}' не знайдено! "
            f"Покладіть готовий файл '{file_path}' у папку зі скриптом."
        )

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_enriched{ext}"
    cache = LocalCache()
    tracker = CostTracker()

    if not resume_checkpoint or not os.path.exists(output_path):
        shutil.copyfile(file_path, output_path)

    print(f"📂 Читання початкового файлу: '{file_path}'...")
    df_source = pd.read_excel(file_path)

    print(f"🧠 Аналіз структури та побудова плану збагачення...")
    batch_plan = create_enrichment_plan(df_source, task_description, tracker)
    print(f"📋 Цільових колонок для обробки: {len(batch_plan.tasks)}")

    for task_num, col_task in enumerate(batch_plan.tasks, start=1):
        print(f"\n--- [Колонка {task_num}/{len(batch_plan.tasks)}] '{col_task.target_column}' (тип: {col_task.expected_type}) ---")
        print(f"   ├─ Основний шаблон: '{col_task.search_query_template}'")
        print(f"   └─ Резервний шаблон: '{col_task.fallback_query_template}'")

        target_col_idx = ensure_column_in_workbook(output_path, col_task.target_column)
        df_current = pd.read_excel(output_path)
        actual_col_name = df_current.columns[target_col_idx - 1]

        pending_indices = []
        for idx, row in df_current.iterrows():
            val = row.get(actual_col_name)
            if pd.isna(val) or str(val).strip() == "":
                pending_indices.append(idx)

        if not pending_indices:
            print(f"   ✨ Усі рядки для колонки '{col_task.target_column}' вже заповнені!")
            continue

        print(f"   🚀 Обробка {len(pending_indices)} рядків -> запис у '{output_path}'...")

        for chunk_start in range(0, len(pending_indices), CHECKPOINT_INTERVAL):
            chunk_indices = pending_indices[chunk_start : chunk_start + CHECKPOINT_INTERVAL]

            # КРОК А: Первинний пакетний пошук у Serper
            primary_queries = {
                idx: format_query(col_task.search_query_template, df_current.loc[idx].to_dict(), col_task.extraction_instruction)
                for idx in chunk_indices
            }
            snippets_map = fetch_serper_batches(list(primary_queries.values()), cache, tracker)

            # КРОК Б: Первинна паралельна екстракція через ThreadPoolExecutor
            chunk_results: Dict[int, ExtractionResult] = {}
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_idx = {
                    executor.submit(
                        extract_with_verification,
                        primary_queries[idx],
                        snippets_map.get(primary_queries[idx], ""),
                        col_task,
                        cache,
                        tracker
                    ): idx
                    for idx in chunk_indices
                }
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=f"   Заповнення '{col_task.target_column}'"):
                    idx = future_to_idx[future]
                    chunk_results[idx] = future.result()

            # КРОК В: Гібридна верифікація (Fallback)
            fallback_indices = [
                idx for idx, res in chunk_results.items()
                if res.value is None or res.confidence == "low"
            ]

            if fallback_indices:
                print(f"   🔄 Запуск Fallback-пошуку для {len(fallback_indices)} рядків...")
                fb_queries = {
                    idx: format_query(col_task.fallback_query_template, df_current.loc[idx].to_dict(), col_task.extraction_instruction)
                    for idx in fallback_indices
                }
                fb_snippets_map = fetch_serper_batches(list(fb_queries.values()), cache, tracker)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    fb_future_to_idx = {
                        executor.submit(
                            extract_with_verification,
                            fb_queries[idx],
                            fb_snippets_map.get(fb_queries[idx], ""),
                            col_task,
                            cache,
                            tracker
                        ): idx
                        for idx in fallback_indices
                    }
                    for future in as_completed(fb_future_to_idx):
                        idx = fb_future_to_idx[future]
                        fb_res = future.result()
                        if fb_res.value is not None and (chunk_results[idx].value is None or fb_res.confidence in ("high", "medium")):
                            chunk_results[idx] = fb_res

            # КРОК Г: Запис заповнених даних у новий файл *_enriched.xlsx
            save_checkpoint_to_excel(output_path, target_col_idx, chunk_results)

    df_final = pd.read_excel(output_path)
    print(f"\n✅ Готово! Створено збагачений файл: {output_path}")
    print("📊 Попередній перегляд результату:")
    print(df_final.to_string(index=False))
    tracker.print_summary()
    print("-" * 65)

    return output_path, tracker


def process_excel(file_path: str, task_description: str, resume_checkpoint: bool = False) -> str:
    """Простий інтерфейс виклику згідно з ТЗ."""
    output_path, _ = process_excel_with_stats(file_path, task_description, resume_checkpoint)
    return output_path


# ==========================================
# 9. FASTAPI ВЕБ-СЕРВЕР
# ==========================================
app = FastAPI(
    title="Intelligent Excel Data Enricher API",
    description="REST API для автоматичного збагачення Excel-файлів через LLM (OpenRouter) та Google Search (Serper).",
    version="1.0.0"
)


@app.post("/enrich", summary="Збагатити Excel-файл за текстовим завданням")
def enrich_excel_endpoint(
    file: UploadFile = File(..., description="Вхідний Excel-файл (.xlsx)"),
    task_description: str = Form(..., description="Текстовий опис завдання")
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Підтримуються лише файли формату Excel (.xlsx, .xls)")

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        output_path, tracker = process_excel_with_stats(input_path, task_description)
        enriched_filename = os.path.basename(output_path)

        headers = {
            "X-Total-Cost-USD": f"{tracker.total_cost_usd:.6f}",
            "X-LLM-Tokens": str(tracker.prompt_tokens + tracker.completion_tokens),
            "X-Serper-Queries": str(tracker.serper_queries),
            "X-Cache-Hits": str(tracker.serper_cache_hits + tracker.llm_cache_hits)
        }

        return FileResponse(
            path=output_path,
            filename=enriched_filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 10. CLI ІНТЕРФЕЙС ТА ENTRY POINT
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="🚀 Інтелектуальне збагачення Excel-даних (CLI & FastAPI)"
    )
    parser.add_argument("-f", "--file", type=str, help="Шлях до вхідного Excel-файлу (напр. capitals.xlsx)")
    parser.add_argument("-t", "--task", type=str, help="Текстовий опис завдання у лапках")
    parser.add_argument("--resume", action="store_true", help="Продовжити з останнього чекпоінта")
    parser.add_argument("--serve", action="store_true", help="Запустити FastAPI веб-сервер")
    parser.add_argument("--port", type=int, default=8000, help="Порт для FastAPI сервера (за замовчуванням 8000)")

    args = parser.parse_args()

    if args.serve:
        print(f"🌐 Запуск FastAPI сервера (Порт: {args.port}, Swagger UI: /docs)")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if args.file and args.task:
        process_excel(
            file_path=args.file,
            task_description=args.task,
            resume_checkpoint=args.resume
        )
        return

    print("ℹ️ Аргументи CLI не передані — виконуємо стандартний прогін на готових файлах:\n")

    if os.path.exists("capitals.xlsx"):
        process_excel(
            file_path="capitals.xlsx",
            task_description="знайди пряму відстань між столицями в км для колонки distance"
        )
    else:
        print("⚠️ Файл 'capitals.xlsx' не знайдено у поточній папці.")

    if os.path.exists("mountains.xlsx"):
        process_excel(
            file_path="mountains.xlsx",
            task_description="додай висоту гір у метрах до колонки height"
        )
    else:
        print("⚠️ Файл 'mountains.xlsx' не знайдено у поточній папці.")


if __name__ == "__main__":
    main()