"""
Análise e visualização dos resultados de avaliação do Agente 1746.
Lê os CSVs/JSONs gerados por run_evaluation.py e run_comparison.py
e produz tabelas e gráficos para o capítulo de resultados.

Uso:
  python analyze_results.py [--eval-id ID] [--cmp-id ID]

Se os IDs não forem fornecidos, usa os arquivos mais recentes na pasta results/.
"""

import json
import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def find_latest(prefix: str, ext: str) -> Path | None:
    """Encontra o arquivo mais recente com o prefixo dado."""
    candidates = sorted(RESULTS_DIR.glob(f"{prefix}*{ext}"), reverse=True)
    return candidates[0] if candidates else None


def load_evaluation(eval_id: str | None) -> tuple[pd.DataFrame | None, dict | None]:
    if eval_id:
        details_path = RESULTS_DIR / f"evaluation_details_{eval_id}.csv"
        metrics_path = RESULTS_DIR / f"evaluation_metrics_{eval_id}.json"
    else:
        details_path = find_latest("evaluation_details_", ".csv")
        metrics_path = find_latest("evaluation_metrics_", ".json")

    details = None
    metrics = None

    if details_path and details_path.exists():
        details = pd.read_csv(details_path)
        print(f"Avaliação carregada: {details_path.name}")
    else:
        print("Nenhum resultado de avaliação encontrado.")

    if metrics_path and metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return details, metrics


def load_comparison(cmp_id: str | None) -> tuple[pd.DataFrame | None, dict | None]:
    if cmp_id:
        details_path = RESULTS_DIR / f"comparison_details_{cmp_id}.csv"
        summary_path = RESULTS_DIR / f"comparison_summary_{cmp_id}.json"
    else:
        details_path = find_latest("comparison_details_", ".csv")
        summary_path = find_latest("comparison_summary_", ".json")

    details = None
    summary = None

    if details_path and details_path.exists():
        details = pd.read_csv(details_path)
        print(f"Comparação carregada: {details_path.name}")
    else:
        print("Nenhum resultado de comparação encontrado.")

    if summary_path and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    return details, summary


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def print_precision_table(metrics: dict):
    """Tabela de precisão por categoria."""
    print("\n" + "=" * 55)
    print("PRECISÃO NA GERAÇÃO DE SQL POR CATEGORIA")
    print("=" * 55)
    print(f"{'Categoria':25s} {'Corretos':>10s} {'Total':>8s} {'Precisão':>10s}")
    print("-" * 55)

    by_cat = metrics.get("precisao_sql_por_categoria", {})
    for cat, data in sorted(by_cat.items()):
        print(f"  {cat:23s} {data['corretos']:>10d} {data['total']:>8d} {data['precisao']:>9.1%}")

    print("-" * 55)
    print(f"  {'GERAL':23s} {'':>10s} {'':>8s} {metrics['precisao_sql_geral']:>9.1%}")
    print("=" * 55)


def print_comparison_table(summary: dict):
    """Tabela comparativa RAG vs Fallback."""
    print("\n" + "=" * 65)
    print("COMPARAÇÃO RAG vs FALLBACK")
    print("=" * 65)
    labels = {
        "latencia_media_s": "Latência média (s)",
        "latencia_desvio_s": "Latência σ (s)",
        "tokens_total_medio": "Tokens total (média)",
        "tokens_total_desvio": "Tokens total (σ)",
        "tokens_input_medio": "Tokens input (média)",
        "tokens_output_medio": "Tokens output (média)",
        "erros": "Erros",
    }
    print(f"  {'Métrica':35s} {'RAG':>12s} {'Fallback':>12s}")
    print("-" * 65)
    for key, label in labels.items():
        r = summary.get("rag", {}).get(key, "—")
        f = summary.get("fallback", {}).get(key, "—")
        print(f"  {label:35s} {str(r):>12s} {str(f):>12s}")
    print("=" * 65)


def generate_latex_precision(metrics: dict) -> str:
    """Gera tabela LaTeX de precisão por categoria."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Precisão na geração de SQL por categoria de consulta}",
        r"\label{tab:precisao_sql}",
        r"\begin{tabular}{|l|c|c|c|}",
        r"\hline",
        r"\textbf{Categoria} & \textbf{Corretos} & \textbf{Total} & \textbf{Precisão} \\",
        r"\hline",
    ]

    by_cat = metrics.get("precisao_sql_por_categoria", {})
    for cat, data in sorted(by_cat.items()):
        cat_escaped = cat.replace("_", r"\_")
        lines.append(
            f"{cat_escaped} & {data['corretos']} & {data['total']} & {data['precisao']:.1%} \\\\"
        )
        lines.append(r"\hline")

    lines.append(
        f"\\textbf{{Geral}} & & & \\textbf{{{metrics['precisao_sql_geral']:.1%}}} \\\\"
    )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_latex_comparison(summary: dict) -> str:
    """Gera tabela LaTeX comparativa RAG vs Fallback."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Comparação de desempenho: Banco Vetorial (RAG) vs Fallback (BigQuery)}",
        r"\label{tab:rag_vs_fallback}",
        r"\begin{tabular}{|l|c|c|}",
        r"\hline",
        r"\textbf{Métrica} & \textbf{RAG} & \textbf{Fallback} \\",
        r"\hline",
    ]

    rows = [
        ("Latência média (s)", "latencia_media_s"),
        ("Latência $\\sigma$ (s)", "latencia_desvio_s"),
        ("Tokens total (média)", "tokens_total_medio"),
        ("Tokens total ($\\sigma$)", "tokens_total_desvio"),
        ("Tokens input (média)", "tokens_input_medio"),
        ("Tokens output (média)", "tokens_output_medio"),
    ]

    for label, key in rows:
        r = summary.get("rag", {}).get(key, "—")
        f = summary.get("fallback", {}).get(key, "—")
        lines.append(f"{label} & {r} & {f} \\\\")
        lines.append(r"\hline")

    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plot_precision_by_category(metrics: dict):
    """Gráfico de barras de precisão por categoria."""
    by_cat = metrics.get("precisao_sql_por_categoria", {})
    if not by_cat:
        return

    categories = sorted(by_cat.keys())
    values = [by_cat[c]["precisao"] for c in categories]
    labels = [c.replace("_", " ").title() for c in categories]

    fig = go.Figure(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color="#4C72B0",
        text=[f"{v:.0%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Precisão na Geração de SQL por Categoria",
        xaxis=dict(title="Precisão", range=[0, 1.1]),
        margin=dict(l=150),
        width=900,
        height=450,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(FIGURES_DIR / "precisao_por_categoria.png", scale=2)
    print(f"  Gráfico salvo: {FIGURES_DIR / 'precisao_por_categoria.png'}")


def plot_comparison_latency(summary: dict):
    """Gráfico de barras comparando latência RAG vs Fallback."""
    if not summary:
        return

    modes = ["RAG", "Fallback"]
    means = [summary["rag"]["latencia_media_s"], summary["fallback"]["latencia_media_s"]]
    stds = [summary["rag"]["latencia_desvio_s"], summary["fallback"]["latencia_desvio_s"]]

    fig = go.Figure(go.Bar(
        x=modes,
        y=means,
        error_y=dict(type="data", array=stds),
        marker_color=["#4C72B0", "#DD8452"],
        text=[f"{v:.2f}s" for v in means],
        textposition="outside",
    ))
    fig.update_layout(
        title="Latência Média: RAG vs Fallback",
        yaxis_title="Latência (s)",
        width=550,
        height=400,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(FIGURES_DIR / "comparacao_latencia.png", scale=2)
    print(f"  Gráfico salvo: {FIGURES_DIR / 'comparacao_latencia.png'}")


def plot_comparison_tokens(summary: dict):
    """Gráfico de barras comparando consumo de tokens RAG vs Fallback."""
    if not summary:
        return

    categories = ["Input", "Output", "Total"]
    rag_vals = [
        summary["rag"]["tokens_input_medio"],
        summary["rag"]["tokens_output_medio"],
        summary["rag"]["tokens_total_medio"],
    ]
    fb_vals = [
        summary["fallback"]["tokens_input_medio"],
        summary["fallback"]["tokens_output_medio"],
        summary["fallback"]["tokens_total_medio"],
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="RAG",
        x=categories,
        y=rag_vals,
        marker_color="#4C72B0",
        text=[str(int(v)) for v in rag_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Fallback",
        x=categories,
        y=fb_vals,
        marker_color="#DD8452",
        text=[str(int(v)) for v in fb_vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Consumo de Tokens: RAG vs Fallback",
        yaxis_title="Tokens",
        barmode="group",
        width=750,
        height=450,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(FIGURES_DIR / "comparacao_tokens.png", scale=2)
    print(f"  Gráfico salvo: {FIGURES_DIR / 'comparacao_tokens.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Análise dos resultados de avaliação")
    parser.add_argument("--eval-id", default=None, help="ID da execução de avaliação (timestamp)")
    parser.add_argument("--cmp-id", default=None, help="ID da execução de comparação (timestamp)")
    args = parser.parse_args()

    eval_details, eval_metrics = load_evaluation(args.eval_id)
    cmp_details, cmp_summary = load_comparison(args.cmp_id)

    if eval_metrics:
        print_precision_table(eval_metrics)

        print(f"\nTaxa de sucesso geral:  {eval_metrics['taxa_sucesso_geral']:.1%}")
        if eval_metrics.get("capacidade_contexto") is not None:
            print(f"Capacidade de contexto: {eval_metrics['capacidade_contexto']:.1%}")
        if eval_metrics.get("eficacia_seguranca") is not None:
            print(f"Eficácia de segurança:  {eval_metrics['eficacia_seguranca']:.1%}")

        print("\nGerando gráficos de avaliação...")
        plot_precision_by_category(eval_metrics)

        latex = generate_latex_precision(eval_metrics)
        latex_path = RESULTS_DIR / "tabela_precisao.tex"
        latex_path.write_text(latex, encoding="utf-8")
        print(f"  Tabela LaTeX salva: {latex_path}")

    if cmp_summary:
        print_comparison_table(cmp_summary)

        print("\nGerando gráficos de comparação...")
        plot_comparison_latency(cmp_summary)
        plot_comparison_tokens(cmp_summary)

        latex = generate_latex_comparison(cmp_summary)
        latex_path = RESULTS_DIR / "tabela_comparacao.tex"
        latex_path.write_text(latex, encoding="utf-8")
        print(f"  Tabela LaTeX salva: {latex_path}")

    if not eval_metrics and not cmp_summary:
        print("\nNenhum resultado encontrado. Execute primeiro:")
        print("  python eval/run_evaluation.py")
        print("  python eval/run_comparison.py")


if __name__ == "__main__":
    main()
