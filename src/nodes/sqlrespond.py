from langchain_core.messages import AIMessage
from ..config import logger
from ..llm import make_llm
from ..models import AgentState

_llm = make_llm()

def response_synthesizer(state: AgentState) -> dict:
    """
    Gera uma resposta em linguagem natural com base nos resultados da consulta.
    """
    logger.info(">> Nó: Sintetizador de Resposta")
    question = state['messages'][-1].content
    query_result = state["query_result"]

    # Se não houver resultado, informe o usuário.
    if not query_result:
        answer = "Não encontrei dados para responder a sua pergunta."
    else:
        prompt = f"""
        Você é um assistente de análise de dados. Sua tarefa é responder à pergunta original do usuário de forma clara e objetiva,
        com base nos dados retornados pelo banco de dados.

        Pergunta Original do Usuário:
        "{question}"

        Dados Retornados (em formato de lista de dicionários Python):
        {query_result}

        Sua Resposta (em português, amigável e direta):
        """
        try:
            answer = _llm.invoke(prompt).content
        except Exception as e:
            logger.error(f"Erro na síntese da resposta: {e}", exc_info=True)
            return {"error": "Falha ao gerar a resposta final."}

    logger.info(f"Resposta Gerada: {answer}")
    return {"messages": [AIMessage(content=answer)], "answer": answer}
