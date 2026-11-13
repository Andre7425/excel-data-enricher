import os
from getpass import getpass
def get_secret(name: str, required: bool = True) -> str:
    # 1) змінна середовища, 2) Colab Secrets, 3) ручне введення
    if os.environ.get(name):
        return os.environ[name]
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    hint = "" if required else " (Enter — пропустити)"
    return getpass(f"Введіть {name}{hint}: ")

os.environ['OPENROUTER_API_KEY'] = get_secret("OPENROUTER_API_KEY")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = get_secret("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = "Final_project_AIagent"

import os
import json
import time
import requests
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. КОНФІГУРАЦІЯ ТА КЛЮЧІ
# ==========================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Використовуємо швидку та дешеву модель для масових запитів (1000 рядків)
MODEL_NAME = "openai/gpt-4o-mini"  # або "google/gemini-2.0-flash-001"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==========================================
# 2. МОДЕЛІ ДАНИХ (СХЕМИ ВІДПОВІДЕЙ LLM)
# ==========================================
class EnrichmentPlan(BaseModel):
    target_column: str = Field(description="Назва колонки, яку потрібно заповнити або створити")
    search_query_template: str = Field(
        description="Шаблон пошукового запиту англійською мовою з назвами колонок у фігурних дужках, напр. 'height of {Mountain} {Country} in meters'"
    )
    extraction_instruction: str = Field(
        description="Чітка інструкція, яке саме значення витягнути і в якому форматі (напр. 'Тільки число (відстань у км) без одиниць виміру')"
    )

# ==========================================
# 3. ДОПОМІЖНІ ФУНКЦІЇ (ПОШУК ТА LLM)
# ==========================================
def search_serper(query: str, retries: int = 3) -> str:
    """Виконує пошук через Serper.dev та повертає текстові сніпети."""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = json.dumps({"q": query, "num": 4})

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            snippets = []
            # 1. Перевіряємо пряму відповідь (Answer Box / Knowledge Graph)
            if "answerBox" in data:
                ab = data["answerBox"]
                snippets.append(f"AnswerBox: {ab.get('answer') or ab.get('snippet') or ''}")
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                snippets.append(f"KnowledgeGraph: {kg.get('description', '')} | Attributes: {kg.get('attributes', {})}")
            
            # 2. Додаємо органічні результати
            for item in data.get("organic", []):
                snippets.append(f"- {item.get('title', '')}: {item.get('snippet', '')}")
                
            return "\n".join(snippets) if snippets else "No results found."
        except Exception as e:
            if attempt == retries - 1:
                return f"Search error: {str(e)}"
            time.sleep(1.5 ** attempt)
    return "No results found."


def create_enrichment_plan(df: pd.DataFrame, task_description: str) -> EnrichmentPlan:
    """Аналізує структуру таблиці та створює план збагачення (викликається 1 раз)."""
    columns = list(df.columns)
    sample_data = df.head(3).to_dict(orient="records")
    
    prompt = f"""
    Ти — архітектор даних. Проаналізуй структуру Excel-файлу та завдання користувача.
    
    Колонки таблиці: {columns}
    Приклад перших рядків: {json.dumps(sample_data, ensure_ascii=False)}
    Завдання користувача: "{task_description}"
    
    Поверни виключно валідний JSON з такими полями:
    - "target_column": точна назва колонки для запису результату (якщо вона вже є в списку колонок — використай її, інакше створи нову логічну назву).
    - "search_query_template": ефективний шаблон пошукового запиту в Google (краще англійською мовою для точності) з підстановкою колонок через {{Назва_Колонки}}.
    - "extraction_instruction": сувора інструкція для парсера, в якому форматі повернути результат (наприклад: "Лише ціле або дробове число в км без тексту").
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    plan_data = json.loads(response.choices[0].message.content)
    return EnrichmentPlan(**plan_data)


def extract_value_with_llm(query: str, search_results: str, instruction: str) -> Optional[str]:
    """Витягує конкретне значення з результатів пошуку."""
    if not search_results or search_results.startswith("Search error") or search_results == "No results found.":
        return None

    prompt = f"""
    Запит: {query}
    Результати пошуку з інтернету:
    {search_results}
    
    Інструкція: {instruction}
    
    Поверни результат у форматі JSON з одним ключем "value". 
    Якщо точних даних немає в результатах пошуку, поверни {{"value": null}}.
    Якщо потрібно число — повертай тип number (або float/int), без одиниць виміру (без 'км', 'm', тощо).
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        res_json = json.loads(response.choices[0].message.content)
        return res_json.get("value")
    except Exception as e:
        return None


# ==========================================
# 4. ОБРОБКА РЯДКА ТА ОСНОВНИЙ ПАЙПЛАЙН
# ==========================================
def process_single_row(index: int, row: pd.Series, plan: EnrichmentPlan) -> tuple[int, any]:
    """Обробляє один рядок таблиці: формує запит -> шукає в Serper -> витягує дані через LLM."""
    # Пропускаємо рядок, якщо значення в цільовій колонці вже заповнене
    if plan.target_column in row and pd.notna(row[plan.target_column]) and str(row[plan.target_column]).strip() != "":
        return index, row[plan.target_column]

    # Формуємо пошуковий запит із шаблону
    try:
        query = plan.search_query_template.format(**row.to_dict())
    except KeyError:
        # Fallback, якщо LLM помилилась з назвою колонки в шаблоні
        row_str = " ".join([f"{k}: {v}" for k, v in row.to_dict().items() if k != plan.target_column])
        query = f"{row_str} {plan.extraction_instruction}"

    # Пошук + Екстракція
    snippets = search_serper(query)
    extracted_value = extract_value_with_llm(query, snippets, plan.extraction_instruction)
    
    return index, extracted_value


def process_excel(file_path: str, task_description: str, max_workers: int = 10) -> str:
    """
    Головний інтерфейс системи збагачення Excel-даних.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    print(f"📂 Завантаження файлу: {file_path}")
    df = pd.read_excel(file_path)
    
    print("🧠 Аналіз структури даних та побудова плану...")
    plan = create_enrichment_plan(df, task_description)
    print(f"   ├─ Цільова колонка: '{plan.target_column}'")
    print(f"   ├─ Шаблон пошуку: '{plan.search_query_template}'")
    print(f"   └─ Інструкція екстракції: '{plan.extraction_instruction}'")

    # Якщо цільової колонки ще немає — створюємо її
    if plan.target_column not in df.columns:
        df[plan.target_column] = None

    # Паралельна обробка рядків (масштабування до 1000 записів)
    results = {}
    print(f"🚀 Запуск паралельного збагачення ({len(df)} рядків, потоків: {max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_row, idx, row, plan): idx 
            for idx, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обробка"):
            idx, val = future.result()
            results[idx] = val

    # Записуємо результати зі збереженням порядку
    for idx, val in results.items():
        df.at[idx, plan.target_column] = val

    # Збереження результату
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_enriched{ext}"
    df.to_excel(output_path, index=False)
    print(f"✅ Готово! Збагачений файл збережено: {output_path}\n")
    
    return output_path


# ==========================================
# 5. ТЕСТОВИЙ ЗАПУСК (СТВОРЕННЯ ТА ОБРОБКА)
# ==========================================
if __name__ == "__main__":
    # Створюємо тестові файли з умови завдання, якщо їх немає локально
    capitals_data = pd.DataFrame({
        "Capital_From": ["Kyiv", "London", "Paris", "Berlin", "Warsaw"],
        "Country_From": ["Ukraine", "United Kingdom", "France", "Germany", "Poland"],
        "Capital_To": ["Paris", "Rome", "Berlin", "Vienna", "Kyiv"],
        "Country_To": ["France", "Italy", "Germany", "Austria", "Ukraine"],
        "distance": [None, None, None, None, None]
    })
    capitals_data.to_excel("capitals.xlsx", index=False)

    mountains_data = pd.DataFrame({
        "Mountain": ["Everest", "Mont Blanc", "Denali", "Kilimanjaro"],
        "Country": ["Nepal", "France", "USA", "Tanzania"],
        "height": [None, None, None, None]
    })
    mountains_data.to_excel("mountains.xlsx", index=False)

    # Запуск Тесту 1
    process_excel(
        file_path="capitals.xlsx",
        task_description="знайди пряму відстань між столицями в км для колонки distance"
    )

    # Запуск Тесту 2
    process_excel(
        file_path="mountains.xlsx",
        task_description="додай висоту гір у метрах до колонки height"
    )