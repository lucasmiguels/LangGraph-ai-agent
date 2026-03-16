"""
Script de comparação RAG (ChromaDB) vs Fallback (BigQuery) para o Agente 1746.
Conforme seção 4.2.2 da metodologia.

Para cada consulta com filtro categórico, executa o agente duas vezes:
  - USE_VECTOR_DB = True  (modo RAG)
  - USE_VECTOR_DB = False (modo Fallback)

Métricas coletadas:
  1. Tempo de resposta (total e do nó de categorias)
  2. Contexto de categorias retornado (para avaliação manual de precisão)
  3. Consumo de tokens (via callback do LangChain)
"""

import json
import time
import uuid
import re
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_community.callbacks import get_openai_callback
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import build_graph
from src import config as app_config
from src.config import logger

EVAL_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = EVAL_DIR / "test_cases.json"
RESULTS_DIR = EVAL_DIR / "results"

for _h in logger.handlers[:]:
    if isinstance(_h, logging.FileHandler):
        logger.removeHandler(_h)
_fh = logging.FileHandler(RESULTS_DIR / "eval.log", mode="a", encoding="utf-8")
_fh.setLevel(logging.INFO)
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"))
logger.addHandler(_fh)

CATEGORICAL_COLUMNS = ["tipo", "categoria", "subtipo"]


def extract_category_filters(sql: str) -> tuple[str, str]:
    """Extrai coluna e valor dos filtros categóricos presentes no SQL gerado."""
    if not sql:
        return "", ""
    columns = []
    values = []
    alias = r"(?:\w+\.)?"
    for col in CATEGORICAL_COLUMNS:
        patterns = [
            rf"LOWER\({alias}{col}\)\s*=\s*LOWER\(['\"](.+?)['\"]\)",
            rf"LOWER\({alias}{col}\)\s*=\s*['\"](.+?)['\"]",
            rf"{alias}{col}\s*=\s*['\"](.+?)['\"]",
            rf"{alias}{col}\s+LIKE\s+['\"](.+?)['\"]",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, sql, re.IGNORECASE):
                columns.append(col)
                values.append(match.group(1))
    return "; ".join(columns), "; ".join(values)


def load_categorical_cases() -> list[dict]:
    """Carrega apenas os casos com filtros categóricos."""
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        c for c in data.get("single_turn", [])
        if c["category"] == "filtro_categorico"
    ]


def run_with_mode(compiled_graph, question: str, mode: str, thread_id: str) -> dict:
    """
    Executa uma pergunta no grafo com o modo especificado.
    mode: 'rag' ou 'fallback'
    """
    app_config.USE_VECTOR_DB = (mode == "rag")

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=question)]}

    result = {
        "mode": mode,
        "generated_sql": "",
        "selected_category_column": "",
        "selected_category_value": "",
        "error": "",
        "latency_total_s": 0.0,
        "tokens_input": 0,
        "tokens_output": 0,
        "tokens_total": 0,
    }

    start = time.time()
    try:
        with get_openai_callback() as cb:
            final_state = compiled_graph.invoke(inputs, config=config)

        result["latency_total_s"] = round(time.time() - start, 3)
        result["generated_sql"] = final_state.get("sql_query", "")
        result["error"] = final_state.get("error", "")
        result["tokens_input"] = cb.prompt_tokens
        result["tokens_output"] = cb.completion_tokens
        result["tokens_total"] = cb.total_tokens

        col, val = extract_category_filters(result["generated_sql"])
        result["selected_category_column"] = col
        result["selected_category_value"] = val

    except Exception as e:
        result["latency_total_s"] = round(time.time() - start, 3)
        result["error"] = str(e)

    return result


def save_results(all_results: list[dict], summary: dict, run_id: str):
    """Salva resultados detalhados e resumo comparativo."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = RESULTS_DIR / f"comparison_details_{run_id}.csv"
    fieldnames = [
        "id", "category", "question", "mode",
        "generated_sql", "selected_category_column", "selected_category_value",
        "error", "latency_total_s", "tokens_input", "tokens_output", "tokens_total",
    ]
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    summary_path = RESULTS_DIR / f"comparison_summary_{run_id}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nResultados salvos em:")
    print(f"  Detalhes:  {details_path}")
    print(f"  Resumo:    {summary_path}")


def calculate_summary(all_results: list[dict]) -> dict:
    """Calcula estatísticas comparativas entre RAG e Fallback."""
    import statistics

    summary = {}
    for mode in ("rag", "fallback"):
        mode_results = [r for r in all_results if r["mode"] == mode]
        latencies = [r["latency_total_s"] for r in mode_results]
        tokens = [r["tokens_total"] for r in mode_results]
        tokens_input = [r["tokens_input"] for r in mode_results]
        tokens_output = [r["tokens_output"] for r in mode_results]

        summary[mode] = {
            "n": len(mode_results),
            "latencia_media_s": round(statistics.mean(latencies), 3) if latencies else 0,
            "latencia_desvio_s": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
            "latencia_min_s": round(min(latencies), 3) if latencies else 0,
            "latencia_max_s": round(max(latencies), 3) if latencies else 0,
            "tokens_total_medio": round(statistics.mean(tokens)) if tokens else 0,
            "tokens_total_desvio": round(statistics.stdev(tokens)) if len(tokens) > 1 else 0,
            "tokens_input_medio": round(statistics.mean(tokens_input)) if tokens_input else 0,
            "tokens_output_medio": round(statistics.mean(tokens_output)) if tokens_output else 0,
            "erros": sum(1 for r in mode_results if r["error"]),
        }

    return summary


def print_summary(summary: dict):
    """Exibe um resumo comparativo no terminal."""
    print("\n" + "=" * 70)
    print("COMPARAÇÃO RAG vs FALLBACK")
    print("=" * 70)
    print(f"{'Métrica':35s} {'RAG':>15s} {'Fallback':>15s}")
    print("-" * 70)

    for metric, label in [
        ("n", "Consultas executadas"),
        ("latencia_media_s", "Latência média (s)"),
        ("latencia_desvio_s", "Latência desvio padrão (s)"),
        ("latencia_min_s", "Latência mínima (s)"),
        ("latencia_max_s", "Latência máxima (s)"),
        ("tokens_total_medio", "Tokens total (média)"),
        ("tokens_total_desvio", "Tokens total (desvio)"),
        ("tokens_input_medio", "Tokens input (média)"),
        ("tokens_output_medio", "Tokens output (média)"),
        ("erros", "Erros"),
    ]:
        rag_val = summary["rag"].get(metric, "—")
        fb_val = summary["fallback"].get(metric, "—")
        print(f"  {label:33s} {str(rag_val):>15s} {str(fb_val):>15s}")

    print("=" * 70)
    print("\nNOTA: Verifique 'selected_category_column' e 'selected_category_value'")
    print("no CSV de detalhes para avaliar a precisão na busca de categorias.\n")


def main():
    print("Carregando casos categóricos para comparação...")
    cases = load_categorical_cases()
    print(f"Total de casos: {len(cases)}\n")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = RESULTS_DIR / f"comparison_memory_{run_id}.sqlite"

    all_results = []

    with SqliteSaver.from_conn_string(str(db_path)) as memory:
        compiled_graph = build_graph().compile(checkpointer=memory)
        print("Grafo compilado. Iniciando comparação...\n")

        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case['question'][:65]}...")

            for mode in ("rag", "fallback"):
                thread_id = f"cmp-{case['id']}-{mode}-{uuid.uuid4().hex[:8]}"
                result = run_with_mode(compiled_graph, case["question"], mode, thread_id)
                result["id"] = case["id"]
                result["category"] = case["category"]
                result["question"] = case["question"]

                status = "OK" if not result["error"] else "ERR"
                print(f"    {mode:10s} -> {status} | {result['latency_total_s']:.1f}s | {result['tokens_total']} tokens")
                all_results.append(result)

    app_config.USE_VECTOR_DB = True

    summary = calculate_summary(all_results)
    save_results(all_results, summary, run_id)
    print_summary(summary)


if __name__ == "__main__":
    main()
