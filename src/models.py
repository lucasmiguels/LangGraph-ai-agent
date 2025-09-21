from typing import TypedDict, Literal, List, Dict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    """
    Define a estrutura do estado que flui através do grafo.
    """
    messages: Annotated[list, add_messages]
    plan: Literal["sql_direct", "sql_contextual", "chat"]
    sql_query: str
    schema: str
    category_context: str
    query_result: List[Dict]
    answer: str
    error: str

class IntentRouter(BaseModel):
    """
    Define a estrutura de saída para o roteador de intenção.
    """
    plan: Literal["sql_direct", "sql_contextual", "chat"] = Field(
        description="""
        A decisão sobre qual caminho seguir.
        - 'sql_direct': Use se a pergunta for determinística sobre chamados do 1746, contagens, bairros e não possui filtros de texto (ex: "quantos chamados em 2023?").
        - 'sql_contextual': Use para perguntas sobre chamados do 1746, contagens, bairros, subtipos que filtram por categorias de texto (ex: "qual o subtipo mais frequente de 'Estrutura de Imóvel'?", "quais bairros tiveram chamados de 'Reparo de poste fora de prumo'?").
        - 'chat': Use para saudações, perguntas genéricas ou conversas que não requerem dados.
        """
    )

def format_chat_history(messages: List[BaseMessage]) -> str:
    """Formata o histórico de mensagens para ser usado em um prompt."""
    if not messages:
        return ""
    
    history_str = "Histórico da Conversa Anterior:\n"
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_str += f"Usuário: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_str += f"Assistente: {msg.content}\n"
    return history_str
