from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["prediction"],
    template="""A patient has just been diagnosed with: {prediction}

Generate exactly 3 different search queries to find relevant medical information for this patient.
Each query should focus on a different aspect: what it is, symptoms/effects, and what to expect next.
Return only the 3 queries, one per line, no numbering, no extra text."""
)

def rewrite_query(prediction: str) -> list[str]:
    prompt = prompt_template.format(prediction=prediction)

    response = llm.invoke(prompt)

    queries = response.content.strip().split("\n")
    queries = [q.strip() for q in queries if q.strip()]

    print(f"  Rewritten queries: {queries}")
    return queries[:3]