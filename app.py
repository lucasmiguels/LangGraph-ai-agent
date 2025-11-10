import asyncio
import chainlit as cl
from contextlib import ExitStack
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from src.agent import build_graph  
from src.config import logger

SQLITE_PATH = "agent_memory.sqlite"

def _invoke_blocking(text: str, thread_id: str):
    with SqliteSaver.from_conn_string(SQLITE_PATH) as saver:
        app = build_graph().compile(checkpointer=saver)
        inputs = {"messages": [HumanMessage(content=text)]}
        config = {"configurable": {"thread_id": thread_id}}
        return app.invoke(inputs, config=config)

@cl.on_chat_start
async def on_start():
    stack = ExitStack()

    saver: SqliteSaver = stack.enter_context(SqliteSaver.from_conn_string(SQLITE_PATH))

    app = build_graph().compile(checkpointer=saver)
    logger.info("\nGrafo compilado com memória e pronto para uso interativo.")

    cl.user_session.set("stack", stack)
    cl.user_session.set("saver", saver)
    cl.user_session.set("app", app)

    thread_id = f"thread-{cl.context.session.id}"
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(
        content=(
            "Agente 1746 pronto! (memória ativa por sessão)\n"
            f"**thread_id:** `{thread_id}`\n"
            "Trocar thread: `/thread novo_id`"
        )
    ).send()

@cl.on_message
async def on_msg(msg: cl.Message):
    txt = (msg.content or "").strip()

    # Comando para trocar o thread_id
    if txt.lower().startswith("/thread "):
        new_tid = txt.split(" ", 1)[1].strip() or f"thread-{cl.context.session.id}"
        cl.user_session.set("thread_id", new_tid)
        await cl.Message(content=f"✅ thread_id atualizado para `{new_tid}`").send()
        return

    app = cl.user_session.get("app")
    if app is None:
        await cl.Message(content="Agente indisponível nesta sessão. Reabra o chat.").send()
        return

    thread_id = cl.user_session.get("thread_id", f"thread-{cl.context.session.id}")

    try:
        final_state = await asyncio.to_thread(_invoke_blocking, txt, thread_id)
    except Exception as e:
        logger.error(f"Erro durante a execução do agente: {e}", exc_info=True)
        await cl.Message(content="Agente: Desculpe, ocorreu um erro. Veja `agent.log`.").send()
        return

    answer = final_state.get("answer") if final_state else None
    if not answer and final_state and final_state.get("messages"):
        answer = final_state["messages"][-1].get("content")

    await cl.Message(content=answer or "(sem resposta)").send()

@cl.on_chat_end
async def on_end():
    stack: ExitStack = cl.user_session.get("stack")
    if stack:
        try:
            stack.close()
        except Exception:
            pass
