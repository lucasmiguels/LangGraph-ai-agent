from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from src.agent import build_graph
from src.config import logger

def main():
    with SqliteSaver.from_conn_string("agent_memory.sqlite") as memory:
        compiled_graph = build_graph().compile(checkpointer=memory)
        logger.info("\nGrafo compilado com memória e pronto para uso interativo.")

        thread_id = input("Digite um ID para esta conversa (ex: 'conversa-1'): ")
        if not thread_id:
            thread_id = "default-conversation"
        
        config = {"configurable": {"thread_id": thread_id}}
        print(f"Memória carregada para a conversa: '{thread_id}'. Digite 'sair' para terminar.")

        while True:
            user_input = input("Usuário: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("Agente: Até logo!")
                break
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            try:
                final_state = compiled_graph.invoke(inputs, config=config)

                if final_state and final_state.get("answer"):
                    print(f"Agente: {final_state['answer']}")

            except Exception as e:
                logger.error(f"Erro durante a execução do agente: {e}", exc_info=True)
                print(f"Agente: Desculpe, ocorreu um erro. Verifique o arquivo agent.log para detalhes.")

if __name__ == "__main__":
    main()
