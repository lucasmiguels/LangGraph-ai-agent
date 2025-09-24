# Agente 1746

Bem-vindo! Este chat conecta você ao **Agente 1746**, que transforma perguntas em **SQL (BigQuery)** sobre as bases do 1746 e devolve respostas em PT-BR.


## O que o agente faz
- Entende perguntas em linguagem natural.
- Gera **SQL somente leitura** (com validação).
- Executa no **BigQuery** e sintetiza os resultados.
- Fornece opiniões clubistas sempre que necessário

## Tabelas Permitidas
- `datario.adm_central_atendimento_1746.chamado`
- `datario.dados_mestres.bairro`

- Sem DDL/DML, sem múltiplas instruções, sem consultas fora dessas tabelas.

## Como usar
Digite sua pergunta normalmente.  
Para trocar o **thread_id** (memória) desta conversa, use:
`/thread <novo_id>`
- Cada `thread_id` tem **estado próprio** salvo no arquivo **`agent_memory.sqlite`** (LangGraph/SqliteSaver).
- O histórico/estado de cada thread é **persistido** entre mensagens e **sobrevive a reinicializações** (desde que o arquivo permaneça).
- Para “recomeçar do zero”, use um `thread_id` novo (ex.: `/thread teste-2`) 

## Limites & Segurança
- **Somente SELECT/WITH** (leitura).
- Apenas as tabelas listadas acima são permitidas.

## Ajuda
Problemas ou respostas inesperadas? Verifique `agent.log` para mais informações.

Boa análise!


