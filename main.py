from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class Request(BaseModel):
    problem_id: str
    problem: str

@app.post("/solve")
def solve(req: Request):

    prompt = f"""
Solve the arithmetic word problem carefully.

Ignore irrelevant numbers.

Return ONLY valid JSON.

Required schema:
{{
  "reasoning": "string with at least 80 characters",
  "answer": integer
}}

Problem:
{req.problem}
"""

    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    if (
        set(result.keys()) != {"reasoning", "answer"}
        or not isinstance(result["reasoning"], str)
        or len(result["reasoning"]) < 80
        or not isinstance(result["answer"], int)
    ):
        raise ValueError("Invalid response")

    return result