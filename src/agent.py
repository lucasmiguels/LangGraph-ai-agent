import os
import pandas as pd
from typing import TypedDict, Literal, List, Dict, Any
from dotenv import load_dotenv
from google.cloud import bigquery
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    )

BIGQUERY_PROJECT = os.getenv("BIGQUERY_PROJECT")
CHAMADOS_TABLE_FULL_PATH = "datario.adm_central_atendimento_1746.chamado"
BAIRROS_TABLE_FULL_PATH = "datario.dados_mestres.bairro"

CATEGORICAL_COLUMNS = ["tipo", "subtipo", "categoria"]

SIMULATE_EXECUTION = False

print("Configuração inicial completa.")

class AgentState(TypedDict):
    """
    Define a estrutura do estado que flui através do grafo.
    """
    question: str                  
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

def intent_router(state: AgentState) -> dict:
    """
    Decide se a pergunta do usuário requer uma consulta ao banco de dados ou é conversacional.
    """
    print(">> Nó: Roteador de Intenção")
    question = state["question"]
    
    structured_llm = llm.with_structured_output(IntentRouter)
    
    prompt = f"""
    Analise a seguinte pergunta do usuário e decida o plano de ação.

    Pergunta: "{question}"

    Selecione 'sql' se a pergunta for sobre chamados do 1746, contagens, bairros, subtipos, etc.
    Selecione 'chat' para saudações, agradecimentos ou perguntas genéricas não relacionadas aos dados.
    """
    
    try:
        routing_decision = structured_llm.invoke(prompt)
        print(f"   Decisão: {routing_decision.plan}")
        return {"plan": routing_decision.plan}
    except Exception as e:
        print(f"   Erro no roteador: {e}")
        return {"error": "Falha ao decidir o plano de ação."}

def schema_fetcher(state: AgentState) -> dict:
    """
    Busca o esquema (colunas e tipos) das tabelas no BigQuery
    e o formata em uma string para ser usada no prompt do LLM.
    """
    print(">> Nó: Buscador de Esquema (Schema Fetcher)")
    

    table_full_paths = [CHAMADOS_TABLE_FULL_PATH, BAIRROS_TABLE_FULL_PATH]
    
    schema_details = []
    
    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT)
        print(f"   Conectado ao BigQuery para buscar esquema.")
        
        for table_path in table_full_paths:
            table = client.get_table(table_path)
            
            # Formata o nome da tabela
            schema_details.append(f"Tabela: `{table.full_table_id.replace(':', '.')}`")
            
            # Formata cada coluna e seu tipo
            for column in table.schema:
                schema_details.append(f"- {column.name} ({column.field_type})")
            
            schema_details.append("") 

        formatted_schema = "\n".join(schema_details)
        print(f"   Esquema obtido com sucesso:\n{formatted_schema}")
        
        return {"schema": formatted_schema}

    except Exception as e:
        print(f"   Erro ao buscar o esquema: {e}")
        return {"error": f"Não foi possível buscar o esquema das tabelas: {e}"}


def category_fetcher(state: AgentState) -> dict:
    """
    Busca os valores únicos das colunas 'tipo', 'subtipo' e 'categoria'
    para dar contexto ao LLM sobre os filtros de texto corretos.
    """
    print(">> Nó: Buscador de Categorias (Category Fetcher)")
    
    context_details = []
    
    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT)
        print("   Conectado ao BigQuery para buscar categorias.")
        
        for column in CATEGORICAL_COLUMNS:
            # Esta consulta busca os valores distintos da coluna especificada.
            query = f"SELECT DISTINCT {column} FROM `{CHAMADOS_TABLE_FULL_PATH}` WHERE {column} IS NOT NULL ORDER BY {column}"
            
            results = client.query(query).to_dataframe()
            
            # Formata os valores em uma lista de strings para o prompt
            values = results[column].tolist()
            context_details.append(f"Valores possíveis para a coluna '{column}':\n{values}\n")

        formatted_context = "\n".join(context_details)
        print("   Contexto de categorias obtido com sucesso.")
        
        return {"category_context": formatted_context}

    except Exception as e:
        print(f"   Erro ao buscar categorias: {e}")
        # Retorna um contexto vazio em caso de erro para não quebrar o fluxo.
        return {"category_context": ""}

def sql_generator(state: AgentState) -> dict:
    """
    Gera uma consulta SQL válida e eficiente para o BigQuery com base na pergunta e no schema.
    """
    print(">> Nó: Gerador de SQL")
    question = state["question"]
    schema = state["schema"]
    category_context = state.get("category_context", "")
    context_section = ""
    if category_context:
        context_section = f"""
        INFORMAÇÕES DE CONTEXTO SOBRE AS CATEGORIAS:
        Use a lista de valores abaixo para encontrar o termo e a coluna corretos para a pergunta do usuário.
        {category_context}
        """

    prompt = f"""
    Sua tarefa é ser um especialista em SQL do Google BigQuery. Seu objetivo principal é gerar uma única consulta SQL que seja **correta e funcional**.

    ESQUEMA DO BANCO DE DADOS:
    {schema}
    {context_section}

    REGRAS ESSENCIAIS:
    1.  **Nomes de Tabela:** SEMPRE use o nome completo da tabela (ex: `projeto.dataset.tabela`) nas cláusulas `FROM` e `JOIN`.

    2.  **Filtros de Texto:** Para buscas em colunas de texto (STRING), SEMPRE use a função `LOWER()` para garantir que a busca não seja sensível a maiúsculas/minúsculas. (ex: `WHERE LOWER(tipo) = 'iluminação pública'`). Além disso, é útil usar LIKE para garantir que a busca seja mais flexível.
        - Em caso de ambiguidade, priorize o uso da coluna subtipo.

    3.  **Filtros de Data:** Para todos os filtros de data, use a coluna `data_inicio`.
        - Para um **dia específico** (ex: "no dia 28/11/2024"), use a função `DATE`. Ex: `WHERE DATE(data_inicio) = '2024-11-28'`.
        - Para **intervalos de datas**, use `DATE(data_inicio) BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`.

    4.  **Seleção de Colunas:** NUNCA use `SELECT *`. Selecione apenas as colunas necessárias para responder à pergunta.

    5.  **Formato de Saída:** Retorne APENAS o código SQL, sem nenhuma explicação ou formatação de markdown.

    PERGUNTA DO USUÁRIO:
    "{question}"

    SQL GERADO:
    """
    
    try:
        sql_query = llm.invoke(prompt).content
        cleaned_sql_query = sql_query.strip().replace("```sql", "").replace("```", "").strip()
        print(f"   SQL Gerado: \n{cleaned_sql_query}")
        
        return {"sql_query": cleaned_sql_query}
    
    except Exception as e:
        print(f"   Erro na geração de SQL: {e}")
        return {"error": "Falha ao gerar a consulta SQL."}
def sql_executor(state: AgentState) -> dict:
    """
    Executa a consulta no BigQuery e retorna o resultado.
    """
    print(">> Nó: Executor de SQL")
    sql_query = state["sql_query"]

    if SIMULATE_EXECUTION:
        print("   --- MODO DE SIMULAÇÃO ATIVADO ---")
        mock_data = [
            {'coluna_1': 'dado_a', 'coluna_2': 123},
            {'coluna_1': 'dado_b', 'coluna_2': 456}
        ]
        print(f"   Resultado Simulado: {mock_data}")
        return {"query_result": mock_data}

    try:
        client = bigquery.Client(project=BIGQUERY_PROJECT)
        print("   Conectado ao BigQuery.")
        query_job = client.query(sql_query)
        
        results = query_job.to_dataframe()
        query_result = results.to_dict('records')
        
        print(f"   Consulta executada com sucesso. {len(query_result)} linhas retornadas.")
        return {"query_result": query_result}
    except Exception as e:
        print(f"   Erro na execução da consulta: {e}")
        return {"error": f"Erro ao executar a consulta no BigQuery: {e}"}

def response_synthesizer(state: AgentState) -> dict:
    """
    Gera uma resposta em linguagem natural com base nos resultados da consulta.
    """
    print(">> Nó: Sintetizador de Resposta")
    question = state["question"]
    query_result = state["query_result"]

    # Se não houver resultado, informe o usuário.
    if not query_result:
        return {"answer": "Não encontrei dados para responder a sua pergunta."}

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
        answer = llm.invoke(prompt).content
        print(f"   Resposta Gerada: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"   Erro na síntese da resposta: {e}")
        return {"error": "Falha ao gerar a resposta final."}

def conversational_responder(state: AgentState) -> dict:
    """
    Gera uma resposta conversacional para perguntas que não requerem acesso a dados.
    """
    print(">> Nó: Resposta Conversacional")
    question = state["question"]
    
    prompt = f"""
    Você é um assistente amigável e flamenguista. Responda à seguinte pergunta do usuário de forma conversacional. Sempre tente ser um pouco clubista.

    Pergunta: "{question}"
    """
    
    try:
        answer = llm.invoke(prompt).content
        print(f"   Resposta Gerada: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"   Erro na resposta conversacional: {e}")
        return {"error": "Falha ao gerar uma resposta."}

# --- Montagem do Grafo ---

workflow = StateGraph(AgentState)

workflow.add_node("intent_router", intent_router)
workflow.add_node("schema_fetcher", schema_fetcher)
workflow.add_node("category_fetcher", category_fetcher)
workflow.add_node("sql_generator", sql_generator)
workflow.add_node("sql_executor", sql_executor)
workflow.add_node("response_synthesizer", response_synthesizer)
workflow.add_node("conversational_responder", conversational_responder)

workflow.set_entry_point("intent_router")

workflow.add_edge("category_fetcher", "sql_generator")
workflow.add_edge("schema_fetcher", "sql_generator")
workflow.add_edge("sql_generator", "sql_executor")
workflow.add_edge("sql_executor", "response_synthesizer")
workflow.add_edge("response_synthesizer", END)
workflow.add_edge("conversational_responder", END)

def route_after_intent(state: AgentState):
    """Decide para onde ir após a intenção inicial."""
    print("   Avaliando rota principal...")
    if "error" in state and state["error"]: return END
    
    plan = state.get("plan")
    if plan == "chat":
        return "conversational_responder"
    else: # Para qualquer tipo de SQL, o primeiro passo é sempre pegar o esquema
        return "schema_fetcher"

workflow.add_conditional_edges(
    "intent_router",
    route_after_intent,
    {
        "conversational_responder": "conversational_responder",
        "schema_fetcher": "schema_fetcher"
    }
)

def decide_after_schema(state: AgentState):
    """
    Depois de pegar o esquema, decide se precisa buscar o contexto
    das categorias ou se pode ir direto para a geração de SQL.
    """
    print("   Avaliando rota secundária (pós-esquema)...")
    if "error" in state and state["error"]: return END

    plan = state.get("plan")
    if plan == "sql_contextual":
        print("   Rota: Contextual -> Buscador de Categorias")
        return "category_fetcher"
    else: 
        # Se não, vai direto para a geração do SQL
        print("   Rota: Direta -> Gerador de SQL")
        return "sql_generator"

workflow.add_conditional_edges(
    "schema_fetcher",
    decide_after_schema,
    {
        "category_fetcher": "category_fetcher",
        "sql_generator": "sql_generator"
    }
)

app = workflow.compile()

print("\nGrafo compilado e pronto para uso.")

# --- Função para Execução e Teste ---

def run_agent(question: str):
    """
    Executa o agente com uma pergunta e imprime o resultado final.
    """
    print(f"\n=================================================")
    print(f" EXECUTANDO AGENTE PARA A PERGUNTA: '{question}'")
    print(f"=================================================\n")
    
    inputs = {"question": question}
    final_state = app.invoke(inputs)
    
    print("\n-------------------------------------------------")
    if "error" in final_state and final_state["error"]:
        print(f" ERRO: {final_state['error']}")
    else:
        print(f" RESPOSTA FINAL:\n{final_state['answer']}")
    print("-------------------------------------------------\n")

# --- Perguntas para Teste ---

if __name__ == "__main__":
    # Pergunta de Análise Simples 1
    run_agent("Quantos chamados foram abertos no dia 28/11/2024?")

    # Pergunta de Análise Simples 2
    run_agent("Qual o subtipo de chamado mais comum relacionado a 'Iluminação Pública'?")

    # Pergunta com JOIN e Agregação 
    run_agent("Quais os 3 bairros que mais tiveram chamados abertos sobre 'reparo de buraco' em 2023?")
    
    # Pergunta com Agregação 
    run_agent("Qual o nome da unidade organizacional que mais atendeu chamados de 'Fiscalização de estacionamento irregular'?")

    # Pergunta Conversacional 1
    run_agent("Olá, tudo bem?")

    # Pergunta Conversacional 2
    run_agent("Me dê sugestões de brincadeiras para fazer com meu cachorro!")