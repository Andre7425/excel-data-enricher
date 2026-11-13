import os
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
MAX_CONCURRENT_LLM = 25

# Кольори для маркування комірок в Excel
FILL_LOW_CONFIDENCE = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # Жовтий
FILL_MISSING = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")         # Рожевий


# ==========================================
# 2. МОДЕЛІ ДАНИХ (PYDANTIC)
# ==========================================
class EnrichmentPlan(BaseModel):
    target_column: str = Field(description="Точна назва колонки для запису результату")
    search_query_template: str = Field(description="Основний шаблон запиту англійською з {Назва_Колонки}")
    fallback_query_template: str = Field(description="Альтернативний (ширший) шаблон запиту англійською, якщо перший не дав точних даних")
    extraction_instruction: str = Field(description="Сувора інструкція, яке значення та в яких одиницях витягнути")
    expected_type: Literal["number", "string"] = Field(description="Тип очікуваного значення: 'number' або 'string'")


class ExtractionResult(BaseModel):
    value: Optional[Any] = Field(default=None, description="Витягнуте значення (число або рядок) або null")
    confidence: Literal["high", "medium", "low"] = Field(default="low", description="Рівень впевненості у точності даних")
    unit_verified: bool = Field(default=False, description="Чи відповідає одиниця виміру інструкції")


# ==========================================
# 3. ЛОКАЛЬНИЙ КЕШ ТА ДЕДУПЛІКАЦІЯ
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
            return row[0] if row else None

    def set_search_bulk(self, items: Dict[str, str]):
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO search_cache (query, snippets) VALUES (?, ?)",
                [(self.normalize(q), s) for q, s in items.items()]
            )

    def get_extraction(self, query: str, instruction: str) -> Optional[ExtractionResult]:
        key = self.normalize(f"{query}||{instruction}")
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT result_json FROM extraction_cache WHERE cache_key = ?", (key,))
            row = cur.fetchone()
            if row:
                return ExtractionResult(**json.loads(row[0]))
            return None

    def set_extraction(self, query: str, instruction: str, result: ExtractionResult):
        key = self.normalize(f"{query}||{instruction}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO extraction_cache (cache_key, result_json) VALUES (?, ?)",
                (key, result.model_dump_json())
            )


# ==========================================
# 4. БАТЧИНГ SERPER API + ASYNCIO
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


async def fetch_serper_batches(queries: List[str], cache: LocalCache) -> Dict[str, str]:
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

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=25.0) as http_client:
        for i in range(0, len(missing_queries), SERPER_BATCH_SIZE):
            batch = missing_queries[i : i + SERPER_BATCH_SIZE]
            payload = [{"q": q, "num": 4} for q in batch]

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
                        for q_str in batch:
                            results[q_str] = f"Search error: {str(e)}"
                    await asyncio.sleep(1.5 ** attempt)

    for q in queries:
        if q not in results:
            results[q] = cache.get_search(q) or "No results found."

    return results


# ==========================================
# 5. ПЛАНУВАЛЬНИК ТА ГІБРИДНА ВЕРИФІКАЦІЯ
# ==========================================
async def create_enrichment_plan(
    df: pd.DataFrame,
    task_description: str,
    aclient: AsyncOpenAI
) -> EnrichmentPlan:
    columns = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")

    prompt = f"""
    Проаналізуй структуру таблиці та завдання користувача.
    Колонки: {columns}
    Приклад даних: {json.dumps(sample_data, ensure_ascii=False)}
    Завдання: "{task_description}"

    Поверни строго JSON з полями:
    - "target_column": точна назва цільової колонки (якщо відповідна колонка вже є у списку {columns}, використай її точну назву; якщо немає — придумай нову логічну назву).
    - "search_query_template": точний шаблон пошуку в Google англійською з {{Назва_Колонки}} (напр. "straight line distance between {{Capital_From}}, {{Country_From}} and {{Capital_To}}, {{Country_To}} in km").
    - "fallback_query_template": альтернативний переформульований шаблон англійською, якщо перший не дасть результату (напр. "air distance {{Capital_From}} to {{Capital_To}} kilometers").
    - "extraction_instruction": детальна вимога до значення та одиниць виміру (напр. "Поверни пряму відстань у кілометрах як число. Якщо вказано милі — переведи в км (1 миля = 1.609 км)").
    - "expected_type": "number" (якщо потрібне число) або "string".
    """

    resp = await aclient.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return EnrichmentPlan(**json.loads(resp.choices[0].message.content))


async def extract_with_verification(
    query: str,
    snippets: str,
    plan: EnrichmentPlan,
    cache: LocalCache,
    semaphore: asyncio.Semaphore,
    aclient: AsyncOpenAI
) -> ExtractionResult:
    cached = cache.get_extraction(query, plan.extraction_instruction)
    if cached is not None:
        return cached

    if not snippets or snippets.startswith("Search error") or snippets == "No results found.":
        return ExtractionResult(value=None, confidence="low", unit_verified=False)

    prompt = f"""
    Пошуковий запит: {query}
    Сніпети з пошуку:
    {snippets}

    Інструкція: {plan.extraction_instruction}
    Очікуваний тип: {plan.expected_type}

    Проаналізуй сніпети. Якщо в тексті інша одиниця виміру (наприклад, фути замість метрів або милі замість км), сконвертуй значення математично.
    Поверни JSON:
    {{
        "value": <число або рядок без тексту одиниць виміру, або null якщо даних немає>,
        "confidence": "high" | "medium" | "low",
        "unit_verified": true | false
    }}
    Став "high", якщо цифра прямо підтверджена у сніпетах; "low" — якщо дані суперечливі або відсутні.
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
                data = json.loads(resp.choices[0].message.content)
                result = ExtractionResult(**data)

                if plan.expected_type == "number" and result.value is not None:
                    try:
                        val_str = str(result.value).replace(",", ".").strip()
                        result.value = float(val_str) if "." in val_str else int(val_str)
                    except ValueError:
                        result.confidence = "low"

                cache.set_extraction(query, plan.extraction_instruction, result)
                return result
            except Exception:
                await asyncio.sleep(1.2 ** attempt)

    return ExtractionResult(value=None, confidence="low", unit_verified=False)


# ==========================================
# 6. СТВОРЕННЯ *_enriched.xlsx ТА ЗАПИС КОЛОНОК
# ==========================================
def prepare_workbook(source_path: str, output_path: str, target_column: str, resume: bool = False) -> int:
    """
    Копіює початковий Excel-файл у новий файл із суфіксом _enriched.
    Якщо цільової колонки ще немає — додає її справа.
    Повертає 1-based індекс цільової колонки в openpyxl.
    """
    if not resume or not os.path.exists(output_path):
        shutil.copyfile(source_path, output_path)

    wb = load_workbook(output_path)
    ws = wb.active

    # Шукаємо цільову колонку в першому рядку (заголовках)
    target_col_idx = None
    for col_idx in range(1, ws.max_column + 1):
        cell_val = ws.cell(row=1, column=col_idx).value
        if cell_val and str(cell_val).strip().lower() == target_column.strip().lower():
            target_col_idx = col_idx
            break

    # Якщо такої колонки у вихідному файлі немає — створюємо нову колонку справа
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
    """Точково записує знайдені дані у відповідну колонку нового файлу *_enriched.xlsx."""
    if not updates:
        return

    wb = load_workbook(output_path)
    ws = wb.active

    for df_idx, res in updates.items():
        excel_row = df_idx + 2  # +1 (0-based індекс pandas) +1 (рядок заголовка в Excel)
        cell = ws.cell(row=excel_row, column=target_col_idx)
        cell.value = res.value

        if res.value is None:
            cell.fill = FILL_MISSING
        elif res.confidence == "low" or not res.unit_verified:
            cell.fill = FILL_LOW_CONFIDENCE

    wb.save(output_path)
    wb.close()


# ==========================================
# 7. ГОЛОВНИЙ АСИНХРОННИЙ ПАЙПЛАЙН
# ==========================================
def format_query(template: str, row_dict: Dict[str, Any], instruction: str) -> str:
    try:
        return template.format(**row_dict)
    except KeyError:
        fallback_str = " ".join([f"{k}: {v}" for k, v in row_dict.items() if pd.notna(v)])
        return f"{fallback_str} {instruction}"


async def process_excel_async(file_path: str, task_description: str, resume_checkpoint: bool = False) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ Початковий файл '{file_path}' не знайдено! "
            f"Покладіть готовий файл '{file_path}' у папку зі скриптом."
        )

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_enriched{ext}"
    cache = LocalCache()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)

    async with AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    ) as aclient:

        # 1. Читаємо початковий Excel-файл
        print(f"📂 Читання початкового файлу: '{file_path}'...")
        df_source = pd.read_excel(file_path)

        print(f"🧠 Аналіз структури та побудова плану збагачення...")
        plan = await create_enrichment_plan(df_source, task_description, aclient)
        print(f"   ├─ Цільова колонка: '{plan.target_column}' (тип: {plan.expected_type})")
        print(f"   ├─ Основний шаблон: '{plan.search_query_template}'")
        print(f"   └─ Резервний шаблон: '{plan.fallback_query_template}'")

        # 2. Створюємо новий файл *_enriched.xlsx на базі початкового
        target_col_idx = prepare_workbook(file_path, output_path, plan.target_column, resume=resume_checkpoint)
        df_current = pd.read_excel(output_path)

        # Назва колонки в DataFrame (на випадок різниці регістрів)
        actual_col_name = df_current.columns[target_col_idx - 1]

        # Визначаємо рядки, де цільова колонка порожня
        pending_indices = []
        for idx, row in df_current.iterrows():
            val = row.get(actual_col_name)
            if pd.isna(val) or str(val).strip() == "":
                pending_indices.append(idx)

        if not pending_indices:
            print(f"✨ Усі рядки у '{output_path}' вже заповнені!")
            return output_path

        print(f"🚀 До обробки: {len(pending_indices)} із {len(df_current)} рядків -> запис у '{output_path}'...")

        # 3. Пакетний пошук та паралельне збагачення
        for chunk_start in range(0, len(pending_indices), CHECKPOINT_INTERVAL):
            chunk_indices = pending_indices[chunk_start : chunk_start + CHECKPOINT_INTERVAL]

            # КРОК А: Первинний пакетний пошук у Serper
            primary_queries = {
                idx: format_query(plan.search_query_template, df_current.loc[idx].to_dict(), plan.extraction_instruction)
                for idx in chunk_indices
            }
            snippets_map = await fetch_serper_batches(list(primary_queries.values()), cache)

            # КРОК Б: Первинна паралельна екстракція через LLM
            tasks = [
                extract_with_verification(
                    primary_queries[idx],
                    snippets_map.get(primary_queries[idx], ""),
                    plan,
                    cache,
                    semaphore,
                    aclient
                )
                for idx in chunk_indices
            ]
            extracted_list = await tqdm.gather(*tasks, desc=f"Чанк {chunk_start // CHECKPOINT_INTERVAL + 1} (Основний)")
            chunk_results: Dict[int, ExtractionResult] = dict(zip(chunk_indices, extracted_list))

            # КРОК В: Гібридна верифікація (Fallback для low confidence або null)
            fallback_indices = [
                idx for idx, res in chunk_results.items()
                if res.value is None or res.confidence == "low"
            ]

            if fallback_indices:
                print(f"   🔄 Запуск Fallback-пошуку для {len(fallback_indices)} рядків...")
                fb_queries = {
                    idx: format_query(plan.fallback_query_template, df_current.loc[idx].to_dict(), plan.extraction_instruction)
                    for idx in fallback_indices
                }
                fb_snippets_map = await fetch_serper_batches(list(fb_queries.values()), cache)

                fb_tasks = [
                    extract_with_verification(
                        fb_queries[idx],
                        fb_snippets_map.get(fb_queries[idx], ""),
                        plan,
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
            print(f"   💾 Прогрес збережено у '{output_path}' ({min(chunk_start + CHECKPOINT_INTERVAL, len(pending_indices))}/{len(pending_indices)})")

    print(f"✅ Готово! Створено збагачений файл: {output_path}\n")
    return output_path


def process_excel(file_path: str, task_description: str, resume_checkpoint: bool = False) -> str:
    """
    Головний інтерфейс виклику:
    - Бере готові дані з file_path (напр. 'capitals.xlsx')
    - Створює новий файл із суфіксом _enriched (напр. 'capitals_enriched.xlsx') із заповненими колонками
    """
    return asyncio.run(process_excel_async(file_path, task_description, resume_checkpoint))


# ==========================================
# 8. ЗАПУСК НА ГОТОВИХ ФАЙЛАХ
# ==========================================
if __name__ == "__main__":
    # Система читає ваші готові файли capitals.xlsx та mountains.xlsx з диска
    # і створює нові файли capitals_enriched.xlsx та mountains_enriched.xlsx
    
    process_excel(
        file_path="capitals.xlsx",
        task_description="знайди пряму відстань між столицями в км для колонки distance"
    )

    process_excel(
        file_path="mountains.xlsx",
        task_description="додай висоту гір у метрах до колонки height"
    )