from langchain_openai import ChatOpenAI
from .config import OPENAI_API_KEY

def make_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
