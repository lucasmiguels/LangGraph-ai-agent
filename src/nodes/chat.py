from langchain_core.messages import AIMessage
from ..config import logger
from ..llm import make_llm
from ..models import AgentState, format_chat_history

_llm = make_llm()

def conversational_responder(state: AgentState) -> dict:
    """
    Gera uma resposta conversacional para perguntas que não requerem acesso a dados.
    """
    logger.info(">> Nó: Resposta Conversacional")
    question = state['messages'][-1].content
    chat_history = format_chat_history(state['messages'][:-1])
    
    prompt = f"""
    Você é um assistente amigável e flamenguista. Responda à seguinte pergunta do usuário de forma conversacional. Sempre tente ser um pouco clubista.
    IMPORTANTE: Se for usar emojis, use apenas os que forem das cores vermelho e preto
    {chat_history}
    Pergunta: "{question}"
    """
    
    try:
        answer = _llm.invoke(prompt).content
        logger.info(f"   Resposta Gerada: {answer}")
        return {"messages": [AIMessage(content=answer)], "answer": answer}
    except Exception as e:
        logger.error(f"   Erro na resposta conversacional: {e}")
        return {"error": "Falha ao gerar uma resposta."}
