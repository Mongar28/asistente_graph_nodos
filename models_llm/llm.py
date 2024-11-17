from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


def load_openai_llm(model: str = "gpt-4o-mini",
                    temperature: int = 0,
                    max_tokens: int = 500) -> ChatOpenAI:

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
