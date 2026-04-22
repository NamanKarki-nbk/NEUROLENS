from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(prediction: str) -> list[str]:
    prompt = f"""A patient has just been diagnosed with: {prediction}

Generate exactly 3 different search queries to find relevant medical information for this patient.
Each query should focus on a different aspect: what it is, symptoms/effects, and what to expect next.
Return only the 3 queries, one per line, no numbering, no extra text."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    queries = response.choices[0].message.content.strip().split("\n")
    queries = [q.strip() for q in queries if q.strip()]
    print(f"  Rewritten queries: {queries}")
    return queries[:3]