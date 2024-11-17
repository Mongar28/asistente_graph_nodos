from models_llm.llm import load_openai_llm
from agents.agents import search_agent


llm = load_openai_llm()

# print(
#     llm.invoke("QUe diferencias tiene el polo sur del norte?")
# )
query: str = "Que edad tenía James Rodriguez cuando la seleccioń Colombia gano la copa libertadores?"
response = search_agent(query)
print(response)
