from ..config import logger
from ..llm import make_llm
from ..models import AgentState, IntentRouter, format_chat_history

_llm = make_llm()

def intent_router(state: AgentState) -> dict:
    """
    Decide o plano de ação com base na última pergunta do usuário e no histórico.
    """
    logger.info(">> Nó: Roteador de Intenção")
    
    # A pergunta atual é a última da lista de mensagens
    question = state['messages'][-1].content
    chat_history = format_chat_history(state['messages'][:-1])

    structured_llm = _llm.with_structured_output(IntentRouter)
    
    prompt = f"""
    {chat_history}
    Analise a ÚLTIMA pergunta do usuário abaixo e decida o plano de ação.

    ÚLTIMA Pergunta: "{question}"
    """
    
    try:
        routing_decision = structured_llm.invoke(prompt)
        logger.info(f"   Decisão: {routing_decision.plan}")
        return {"plan": routing_decision.plan}
    except Exception as e:
        logger.info(f"   Erro no roteador: {e}")
        return {"error": "Falha ao decidir o plano de ação."}
