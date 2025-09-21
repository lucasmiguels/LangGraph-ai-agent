from langgraph.graph import StateGraph, END
from .models import AgentState
from .config import logger
from .nodes.intent import intent_router
from .nodes.schema import schema_fetcher
from .nodes.category import category_fetcher
from .nodes.sqlgen import sql_generator
from .nodes.sqlexec import sql_executor
from .nodes.sqlrespond import response_synthesizer
from .nodes.chat import conversational_responder

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    
    graph.add_node("intent_router", intent_router)
    graph.add_node("schema_fetcher", schema_fetcher)
    graph.add_node("category_fetcher", category_fetcher)
    graph.add_node("sql_generator", sql_generator)
    graph.add_node("sql_executor", sql_executor)
    graph.add_node("response_synthesizer", response_synthesizer)
    graph.add_node("conversational_responder", conversational_responder)

    graph.set_entry_point("intent_router")
    graph.add_edge("category_fetcher", "sql_generator")
    graph.add_edge("sql_generator", "sql_executor")
    graph.add_edge("sql_executor", "response_synthesizer")
    graph.add_edge("response_synthesizer", END)
    graph.add_edge("conversational_responder", END)

    def _route_after_intent(state: AgentState):
        """Decide para onde ir após a intenção inicial."""
        logger.info("   Avaliando rota principal...")
        if "error" in state and state["error"]: return END
        
        plan = state.get("plan")
        if plan == "chat":
            return "conversational_responder"
        else: # Para qualquer tipo de SQL, o primeiro passo é sempre pegar o esquema
            return "schema_fetcher"

    graph.add_conditional_edges("intent_router", _route_after_intent, {
        "conversational_responder": "conversational_responder",
        "schema_fetcher": "schema_fetcher",
    })

    def _decide_after_schema(state: AgentState):
        """
        Depois de pegar o esquema, decide se precisa buscar o contexto
        das categorias ou se pode ir direto para a geração de SQL.
        """
        logger.info("   Avaliando rota secundária (pós-esquema)...")
        if "error" in state and state["error"]: return END

        plan = state.get("plan")
        if plan == "sql_contextual":
            logger.info("   Rota: Contextual -> Buscador de Categorias")
            return "category_fetcher"
        else: 
            # Se não, vai direto para a geração do SQL
            logger.info("   Rota: Direta -> Gerador de SQL")
            return "sql_generator"


    graph.add_conditional_edges(
        "schema_fetcher",
        _decide_after_schema,
        {
            "category_fetcher": "category_fetcher",
            "sql_generator": "sql_generator"
        }
    )

    return graph
