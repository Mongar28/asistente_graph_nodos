from langchain.agents import AgentExecutor, create_openai_functions_agent, create_react_agent
from langchain import hub
from dotenv import load_dotenv
from typing import Dict, TypedDict
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from models_llm.llm import load_openai_llm
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,)
# from assistant_graph import State


from langchain.schema import HumanMessage, AIMessage


class State(TypedDict):
    query: str
    category: str
    response: str
    history: list  # Historial de la conversación


def categorize(state: State) -> State:
    """Categorize the customer query into General or Search"""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "General, Search, Email. Return only the category name. Query: {query}"
    )
    llm = load_openai_llm()
    chain = prompt | llm

    # Pasar el historial completo al LLM si es necesario para el contexto
    prompt_with_history = f"{state['history']} Query: {state['query']}"

    category = chain.invoke({"query": prompt_with_history}).content

    # Actualizar historial
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=f"Category: {category}"))

    return {"category": category, "history": state["history"]}


def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    llm = load_openai_llm()
    chain = prompt | llm

    # Pasar el historial completo al LLM si es necesario para el contexto
    prompt_with_history = f"{state['history']} Query: {state['query']}"

    response = chain.invoke({"query": prompt_with_history}).content

    # Actualizar historial
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=response))

    return {"response": response, "history": state["history"]}


def search_agent(state: State) -> State:
    """Build a response to a query with information searched on the internet."""
    search = TavilySearchResults(max_results=5)
    tools = [search]
    llm = load_openai_llm()

    instructions = """
    You are Cristian's virtual assistant, and your task is to search for information on the internet based on the given query
    and provide a clear, concise, and polite response. Ensure that your tone is always friendly, respectful, and attentive."""
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Pasar el historial completo al LLM si es necesario para el contexto
    prompt_with_history = f"{state['history']} Query: {state['query']}"

    agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    response = agent_executor(
        {"input": prompt_with_history}
    )["output"]

    # Actualizar historial
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=response))

    return {'response': response, "history": state["history"]}


@tool
def redact_email(topic: str) -> str:
    """Use this tool to draft the content of an email based on a topic."""

    llm = load_openai_llm()
    # Create prompt for the LLM
    prompt = (
        "Please redact a email based on the topic:\n\n"
        "Topic: {}\n\n"
        "Email Content: [Your email content here]"
    ).format(topic)

    response = llm.invoke(prompt)
    return response


def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    llm = load_openai_llm()
    chain = prompt | llm

    # Pasar el historial completo al LLM si es necesario para el contexto
    prompt_with_history = f"{state['history']} Query: {state['query']}"

    response = chain.invoke({"query": prompt_with_history}).content

    # Actualizar historial
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=response))

    return {"response": response, "history": state["history"]}


def search_agent(state: State) -> State:
    """Build a response to a query with information searched on the internet."""
    search = TavilySearchResults(max_results=5)
    tools = [search]
    llm = load_openai_llm()

    instructions = """
    You are Cristian's virtual assistant, and your task is to search for information on the internet based on the given query
    and provide a clear, concise, and polite response. Ensure that your tone is always friendly, respectful, and attentive."""
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Pasar el historial completo al LLM si es necesario para el contexto
    prompt_with_history = f"{state['history']} Query: {state['query']}"

    agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    response = agent_executor(
        {"input": prompt_with_history}
    )["output"]

    # Actualizar historial
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=response))

    return {'response': response, "history": state["history"]}


def email_egent(state: State) -> State:
    """Build a response to email-related tasks, such as drafting, sending, or reviewing emails."""
    toolkit = GmailToolkit()

    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/"],
        client_secrets_file="credentials.json",
    )
    api_resource = build_resource_service(credentials=credentials)

    toolkit = GmailToolkit(api_resource=api_resource)
    GmailToolkit.model_rebuild()

    tools = toolkit.get_tools()
    tools.append(redact_email)

    llm = load_openai_llm()

    instructions = """
    You are Cristian Montoya's personal assistant."""
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Pasar el historial completo al LLM si es necesario para el contexto
    prompt_with_history = f"{state['history']} Query: {state['query']}"

    agent = create_openai_functions_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    response = agent_executor(
        {"input": prompt_with_history}
    )["output"]

    # Actualizar historial
    state["history"].append(HumanMessage(content=state["query"]))
    state["history"].append(AIMessage(content=response))

    return {'response': response, "history": state["history"]}
