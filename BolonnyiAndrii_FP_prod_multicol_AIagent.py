import os
import re
import json
import sqlite3
import asyncio
import shutil
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import httpx
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

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

MODEL_NAME = "openai/gpt-4o-mini"
CACHE_DB_PATH = "enrichment_cache.sqlite"
SERPER_BATCH_SIZE = 50
CHECKPOINT_INTERVAL = 50
MAX_CONCURRENT_LLM = 5  # Оптимально для WSL2 та уникнення скидання з'єднань Cloudflare

# Кольори для маркування комірок в Excel
FILL_LOW_CONFIDENCE = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # Жовтий
FILL_MISSING = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")         # Рожевий


def create_resilient_http_client() -> httpx.AsyncClient:
    """
    Створює HTTP-клієнт з примусовим IPv4 (local_address='0.0.0.0') та ретраями.
    Вирішує проблему APIConnectionError у середовищі WSL2 / Python 3.14.
    """
    transport = httpx.AsyncHTTPTransport(
        local_address="0.0.0.0",  # Вимикає проблемний IPv6 у WSL2
        retries=3
    )
    return httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(60.0, connect=20.0),
        limits=httpx.Limits(max_connections=15, max_keepalive_connections=10),
        trust_env=False  # Ігнорує системні проксі, які можуть ламати httpx
    )


# ==========================================
# 2. МОДЕЛІ ДАНИХ (ПІДТРИМКА БАГАТЬОХ КОЛОНОК)
# ==========================================
class ColumnTask(BaseModel):
    target_column: str = Field(description="Точна назва ОДНІЄЇ колонки для запису результату")
    search_query_template: str = Field(description="Основний шаблон запиту англійською з {Назва_Колонки}")
    fallback_query_template: str = Field(description="Альтернативний (ширший) шаблон запиту англійською")
    extraction_instruction: str = Field(description="Сувора інструкція, яке значення та в яких одиницях витягнути")
    expected_type: Literal["number", "string"] = Field(description="Тип очікуваного значення: 'number' або 'string'")


class EnrichmentBatchPlan(BaseModel):
    tasks: List[ColumnTask] = Field(description="Список завдань для кожної окремої колонки, яку потрібно заповнити або створити")


class ExtractionResult(BaseModel):
    value: Optional[Any] = Field(default=None, description="Витягнуте значення (число або рядок) або null")
    confidence: Literal["high", "medium", "low"] = Field(default="low", description="Рівень впевненості у точності даних")
    unit_verified: bool = Field(default=False, description="Чи відповідає одиниця виміру інструкції")


# ==========================================
# 3. ДОПОМІЖНІ УТИЛІТИ ПАРСИНГУ JSON І ЧИСЕЛ
# ==========================================
def clean_json_text(text: str) -> str:
    """Очищує відповідь LLM від markdown-блоків ```json ... ```."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_numeric_value(val: Any) -> Optional[float | int]:
    """Перетворює текстові числа (включно з '2.8 million', '1,420 km') у чистий int/float."""
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


# ==========================================
# 4. ЛОКАЛЬНИЙ КЕШ ТА ДЕДУПЛІКАЦІЯ
# ==========================================
class LocalCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
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
        with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO search_cache (query, snippets) VALUES (?, ?)",
                    valid_items
                )

    def get_extraction(self, query: str, instruction: str) -> Optional[ExtractionResult]:
        key = self.normalize(f"{query}||{instruction}")
        with sqlite3.connect(self.db_path) as conn:
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
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO extraction_cache (cache_key, result_json) VALUES (?, ?)",
                (key, result.model_dump_json())
            )


# ==========================================
# 5. БАТЧИНГ SERPER API + ASYNCIO
# ==========================================
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


async def fetch_serper_batches(
    queries: List[str],
    cache: LocalCache,
    http_client: httpx.AsyncClient
) -> Dict[str, str]:
    results: Dict[str, str] = {}
    missing_queries = []

    seen_normalized = set()
    for q in queries:
        cached = cache.get_search(q)
        if cached is not None:
            results[q] = cached
        else:
            norm_q = cache.normalize(q)
            if norm_q not in seen_normalized:
                seen_normalized.add(norm_q)
                missing_queries.append(q)

    if not missing_queries:
        return results

    url = "[https://google.serper.dev/search](https://google.serper.dev/search)"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    for i in range(0, len(missing_queries), SERPER_BATCH_SIZE):
        batch = missing_queries[i : i + SERPER_BATCH_SIZE]
        payload = [{"q": q, "num": 5} for q in batch]

        for attempt in range(3):
            try:
                resp = await http_client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                batch_data = resp.json()

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
                await asyncio.sleep(2.0 ** attempt)

    for q in queries:
        if q not in results:
            results[q] = cache.get_search(q) or "No results found."

    return results


# ==========================================
# 6. МУЛЬТИ-КОЛОНКОВИЙ ПЛАНУВАЛЬНИК ТА ВЕРИФІКАЦІЯ
# ==========================================
async def create_enrichment_plan(
    df: pd.DataFrame,
    task_description: str,
    aclient: AsyncOpenAI
) -> EnrichmentBatchPlan:
    columns = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")

    prompt = f"""
    Проаналізуй структуру Excel-таблиці та завдання користувача.
    Наявні колонки: {columns}
    Приклад перших рядків: {json.dumps(sample_data, ensure_ascii=False)}
    Завдання користувача: "{task_description}"

    ВАЖЛИВО: Якщо користувач просить знайти КІЛЬКА різних показників або заповнити КІЛЬКА колонок (наприклад, і 'population', і 'distance'), створи окремий об'єкт у списку "tasks" для КОЖНОЇ колонки!
    У фігурних дужках шаблонів використовуй ТІЛЬКИ існуючі назви вхідних колонок із {columns}.

    Поверни строго JSON об'єкт із ключем "tasks" (список об'єктів):
    {{
      "tasks": [
        {{
          "target_column": "назва цільової колонки (напр. 'population' або 'distance')",
          "search_query_template": "шаблон пошуку в Google англійською з {{Назва_Колонки}} (напр. 'population of {{Capital_From}}, {{Country_From}} city')",
          "fallback_query_template": "альтернативний переформульований шаблон англійською (напр. '{{Capital_From}} total population inhabitants')",
          "extraction_instruction": "детальна вимога, яке значення повернути (напр. 'Поверни чисельність населення міста як ціле число (напр. 2950000). Якщо потрібно число — тільки число, якщо текст — стислий рядок.')",
          "expected_type": "number" або "string"
        }}
      ]
    }}
    """

    for attempt in range(3):
        try:
            resp = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            raw_content = clean_json_text(resp.choices[0].message.content)
            return EnrichmentBatchPlan(**json.loads(raw_content))
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"❌ Не вдалося з'єднатися з OpenRouter для побудови плану: {e}") from e
            print(f"⏳ Повторна спроба з'єднання з OpenRouter ({attempt + 1}/3)...")
            await asyncio.sleep(2.0 ** attempt)


async def extract_with_verification(
    query: str,
    snippets: str,
    task_plan: ColumnTask,
    cache: LocalCache,
    semaphore: asyncio.Semaphore,
    aclient: AsyncOpenAI
) -> ExtractionResult:
    cached = cache.get_extraction(query, task_plan.extraction_instruction)
    if cached is not None:
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
    - Якщо в сніпетах є приблизні або актуальні дані — обов'язково витягни їх (не повертай null, якщо цифра є у тексті).

    Поверни строго валідний JSON:
    {{
        "value": <число або рядок, або null якщо даних взагалі немає>,
        "confidence": "high" | "medium" | "low",
        "unit_verified": true | false
    }}
    """

    async with semaphore:
        for attempt in range(3):
            try:
                resp = await aclient.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                raw_text = clean_json_text(resp.choices[0].message.content)
                data = json.loads(raw_text)
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
                if attempt == 2:
                    print(f"⚠️ Помилка LLM при обробці запиту '{query}': {e}")
                await asyncio.sleep(2.0 ** attempt)

    return ExtractionResult(value=None, confidence="low", unit_verified=False)


# ==========================================
# 7. РОБОТА З EXCEL ТА ЗАПИС БАГАТЬОХ КОЛОНОК
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
# 8. ГОЛОВНИЙ АСИНХРОННИЙ ПАЙПЛАЙН
# ==========================================
def format_query(template: str, row_dict: Dict[str, Any], instruction: str) -> str:
    clean_dict = {k: ("" if pd.isna(v) else v) for k, v in row_dict.items()}
    try:
        return template.format(**clean_dict).strip()
    except KeyError:
        fallback_str = " ".join([f"{k}: {v}" for k, v in clean_dict.items() if v != ""])
        return f"{fallback_str} {instruction}".strip()


async def process_excel_async(file_path: str, task_description: str, resume_checkpoint: bool = False) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Початковий файл '{file_path}' не знайдено! "
            f"Переконайтесь, що файл лежить у поточній директорії."
        )

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_enriched{ext}"
    cache = LocalCache()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    if not resume_checkpoint or not os.path.exists(output_path):
        shutil.copyfile(file_path, output_path)

    # Використовуємо наш захищений IPv4 HTTP-клієнт як для OpenRouter, так і для Serper
    async with create_resilient_http_client() as custom_http_client:
        async with AsyncOpenAI(
            base_url="[https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)",
            api_key=OPENROUTER_API_KEY,
            http_client=custom_http_client,
            max_retries=3
        ) as aclient:

            print(f"📂 Читання початкового файлу: '{file_path}'...")
            df_source = pd.read_excel(file_path)

            print(f"🧠 Аналіз завдання та побудова плану збагачення...")
            batch_plan = await create_enrichment_plan(df_source, task_description, aclient)
            print(f"📋 Знайдено цільових колонок для обробки: {len(batch_plan.tasks)}")

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

                print(f"   🚀 Обробка {len(pending_indices)} рядків для '{col_task.target_column}'...")

                for chunk_start in range(0, len(pending_indices), CHECKPOINT_INTERVAL):
                    chunk_indices = pending_indices[chunk_start : chunk_start + CHECKPOINT_INTERVAL]

                    # КРОК А: Первинний пакетний пошук у Serper
                    primary_queries = {
                        idx: format_query(col_task.search_query_template, df_current.loc[idx].to_dict(), col_task.extraction_instruction)
                        for idx in chunk_indices
                    }
                    snippets_map = await fetch_serper_batches(list(primary_queries.values()), cache, custom_http_client)

                    # КРОК Б: Первинна паралельна екстракція через LLM
                    tasks = [
                        extract_with_verification(
                            primary_queries[idx],
                            snippets_map.get(primary_queries[idx], ""),
                            col_task,
                            cache,
                            semaphore,
                            aclient
                        )
                        for idx in chunk_indices
                    ]
                    extracted_list = await tqdm.gather(*tasks, desc=f"   Заповнення '{col_task.target_column}'")
                    chunk_results: Dict[int, ExtractionResult] = dict(zip(chunk_indices, extracted_list))

                    # КРОК В: Гібридна верифікація (Fallback)
                    fallback_indices = [
                        idx for idx, res in chunk_results.items()
                        if res.value is None or res.confidence == "low"
                    ]

                    if fallback_indices:
                        print(f"   🔄 Запуск Fallback-пошуку для {len(fallback_indices)} рядків у '{col_task.target_column}'...")
                        fb_queries = {
                            idx: format_query(col_task.fallback_query_template, df_current.loc[idx].to_dict(), col_task.extraction_instruction)
                            for idx in fallback_indices
                        }
                        fb_snippets_map = await fetch_serper_batches(list(fb_queries.values()), cache, custom_http_client)

                        fb_tasks = [
                            extract_with_verification(
                                fb_queries[idx],
                                fb_snippets_map.get(fb_queries[idx], ""),
                                col_task,
                                cache,
                                semaphore,
                                aclient
                            )
                            for idx in fallback_indices
                        ]
                        fb_extracted_list = await asyncio.gather(*fb_tasks)

                        for idx, fb_res in zip(fallback_indices, fb_extracted_list):
                            if fb_res.value is not None and (chunk_results[idx].value is None or fb_res.confidence in ("high", "medium")):
                                chunk_results[idx] = fb_res

                    # КРОК Г: Запис заповнених даних у новий файл *_enriched.xlsx
                    save_checkpoint_to_excel(output_path, target_col_idx, chunk_results)

    df_final = pd.read_excel(output_path)
    print(f"\n✅ Готово! Збагачений файл збережено: '{output_path}'")
    print("📊 Попередній перегляд результату:")
    print(df_final.to_string(index=False))
    print("-" * 60)

    return output_path


def process_excel(file_path: str, task_description: str, resume_checkpoint: bool = False) -> str:
    """Синхронний інтерфейс виклику."""
    return asyncio.run(process_excel_async(file_path, task_description, resume_checkpoint))


# ==========================================
# 9. ТЕСТОВИЙ ЗАПУСК НА ГОТОВИХ ФАЙЛАХ
# ==========================================
if __name__ == "__main__":
    process_excel(
        file_path="capitals.xlsx",
        task_description="додай кількість населення для кожної столиці для колонки population і знайди пряму відстань між столицями в км для колонки distance"
    )

    process_excel(
                file_path="mountains.xlsx",
                task_description="додай GPS координати кожної вершини та висоту гори у метрах до колонок GPS_coordinates, height"
            )