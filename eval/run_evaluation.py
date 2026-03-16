"""
Script de avaliação de corretude do Agente 1746.
Executa os casos de teste definidos em test_cases.json e coleta as métricas
descritas na seção 4.2.1 da metodologia.

Métricas:
  1. Precisão na geração de SQL (por tipo de consulta)
  2. Taxa de sucesso (fluxo sem erros)
"""

import argparse
import json
import time
import uuid
import sys
import csv
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import build_graph
from src.bigquery import get_bq_client
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


def load_test_cases() -> dict:
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def execute_query_on_bigquery(sql: str) -> pd.DataFrame | None:
    """Executa uma query no BigQuery e retorna um DataFrame ordenado."""
    try:
        client = get_bq_client()
        df = client.query(sql).to_dataframe()
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip().str.lower()
        return df
    except Exception as e:
        logger.error(f"Erro ao executar query de validação: {e}")
        return None


def compare_results(expected_df: pd.DataFrame, generated_df: pd.DataFrame) -> bool:
    """Compara dois DataFrames por equivalência semântica (mesmos dados).

    A comparação ignora nomes de colunas (aliases) e verifica apenas se os
    valores retornados são iguais. Quando o DataFrame gerado possui colunas
    extras, tenta todas as combinações de subconjunto com o mesmo número de
    colunas do esperado para encontrar uma correspondência.
    """
    if expected_df is None or generated_df is None:
        return False
    try:
        exp = expected_df.sort_values(by=list(expected_df.columns)).reset_index(drop=True)
        gen = generated_df.sort_values(by=list(generated_df.columns)).reset_index(drop=True)

        if exp.shape[0] != gen.shape[0]:
            return False

        if exp.shape[1] == gen.shape[1]:
            exp_vals = exp.values
            gen_vals = gen.values
            gen_sorted = gen_vals[gen_vals[:, 0].argsort()] if gen_vals.shape[0] > 0 else gen_vals
            exp_sorted = exp_vals[exp_vals[:, 0].argsort()] if exp_vals.shape[0] > 0 else exp_vals
            if pd.DataFrame(exp_sorted).equals(pd.DataFrame(gen_sorted)):
                return True

        if gen.shape[1] > exp.shape[1]:
            from itertools import combinations
            for cols in combinations(range(gen.shape[1]), exp.shape[1]):
                subset = gen.iloc[:, list(cols)]
                subset = subset.sort_values(by=list(subset.columns)).reset_index(drop=True)
                exp_reset = exp.sort_values(by=list(exp.columns)).reset_index(drop=True)
                subset.columns = exp_reset.columns
                if exp_reset.equals(subset):
                    return True

        return False
    except Exception:
        return False


def run_single_turn(compiled_graph, case: dict) -> dict:
    """Executa um caso de teste single-turn e retorna o resultado."""
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=case["question"])]}

    result = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_sql": case.get("expected_sql", ""),
        "generated_sql": "",
        "success": False,
        "sql_correct": False,
        "error": "",
        "latency_s": 0.0,
    }

    start = time.time()
    try:
        final_state = compiled_graph.invoke(inputs, config=config)
        result["latency_s"] = round(time.time() - start, 3)
        result["generated_sql"] = final_state.get("sql_query", "")
        result["error"] = final_state.get("error", "")
        result["success"] = not bool(result["error"])
        result["answer"] = final_state.get("answer", "")
    except Exception as e:
        result["latency_s"] = round(time.time() - start, 3)
        result["error"] = str(e)
        return result

    if result["success"] and result["generated_sql"] and case.get("expected_sql"):
        expected_df = execute_query_on_bigquery(case["expected_sql"])
        generated_df = execute_query_on_bigquery(result["generated_sql"])
        result["sql_correct"] = compare_results(expected_df, generated_df)

    return result


def calculate_metrics(all_results: list[dict]) -> dict:
    """Calcula as métricas de precisão SQL e taxa de sucesso."""
    metrics = {}

    total = len(all_results)
    total_success = sum(1 for r in all_results if r["success"])
    metrics["taxa_sucesso_geral"] = round(total_success / total, 4) if total else 0

    categories = set(r["category"] for r in all_results)
    precision_by_category = {}
    for cat in categories:
        cat_cases = [r for r in all_results if r["category"] == cat]
        correct = sum(1 for r in cat_cases if r["sql_correct"])
        precision_by_category[cat] = {
            "total": len(cat_cases),
            "corretos": correct,
            "precisao": round(correct / len(cat_cases), 4) if cat_cases else 0,
        }
    metrics["precisao_sql_por_categoria"] = precision_by_category

    sql_with_expected = [r for r in all_results if r.get("expected_sql")]
    correct_total = sum(1 for r in sql_with_expected if r["sql_correct"])
    metrics["precisao_sql_geral"] = round(correct_total / len(sql_with_expected), 4) if sql_with_expected else 0

    return metrics


def save_results(all_results: list[dict], metrics: dict, run_id: str):
    """Salva resultados detalhados e métricas agregadas."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = RESULTS_DIR / f"evaluation_details_{run_id}.csv"
    fieldnames = [
        "id", "category", "question", "expected_sql", "generated_sql",
        "success", "sql_correct", "error", "latency_s", "answer",
    ]
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    metrics_path = RESULTS_DIR / f"evaluation_metrics_{run_id}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nResultados salvos em:")
    print(f"  Detalhes: {details_path}")
    print(f"  Métricas: {metrics_path}")


def print_summary(metrics: dict):
    """Exibe um resumo das métricas no terminal."""
    print("\n" + "=" * 60)
    print("RESUMO DA AVALIAÇÃO")
    print("=" * 60)
    print(f"Taxa de sucesso geral:    {metrics['taxa_sucesso_geral']:.1%}")
    print(f"Precisão SQL geral:       {metrics['precisao_sql_geral']:.1%}")

    print("\nPrecisão por categoria:")
    for cat, data in metrics["precisao_sql_por_categoria"].items():
        print(f"  {cat:25s} {data['corretos']}/{data['total']} = {data['precisao']:.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Avaliação de corretude do Agente 1746")
    parser.add_argument("--category", "-c", type=str, default=None,
                        help="Filtrar por categoria (ex: multi_tabela, agregacao, contagem_simples, filtro_data, filtro_categorico)")
    args = parser.parse_args()

    print("Carregando casos de teste...")
    test_data = load_test_cases()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = RESULTS_DIR / f"eval_memory_{run_id}.sqlite"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    with SqliteSaver.from_conn_string(str(db_path)) as memory:
        compiled_graph = build_graph().compile(checkpointer=memory)
        print("Grafo compilado. Iniciando avaliação...\n")

        single_turn_cases = test_data.get("single_turn", [])
        if args.category:
            single_turn_cases = [c for c in single_turn_cases if c["category"] == args.category]
            print(f"Filtrado para categoria '{args.category}': {len(single_turn_cases)} casos\n")
        for i, case in enumerate(single_turn_cases, 1):
            print(f"[{i}/{len(single_turn_cases)}] {case['category']:20s} | {case['question'][:60]}...")
            result = run_single_turn(compiled_graph, case)
            status = "OK" if result["sql_correct"] else "FAIL"
            print(f"         -> {status} ({result['latency_s']:.1f}s)")
            all_results.append(result)

    metrics = calculate_metrics(all_results)
    save_results(all_results, metrics, run_id)
    print_summary(metrics)


if __name__ == "__main__":
    main()
