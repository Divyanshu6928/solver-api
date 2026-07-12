from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import json

client = OpenAI(
    api_key=os.getenv("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openrouter/v1"
)

app = FastAPI()

class Request(BaseModel):
    problem_id: str
    problem: str

@app.post("/solve")
def solve(req: Request):
    try:
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
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Solve arithmetic word problems carefully. "
                    "Ignore irrelevant numbers. "
                    "Return ONLY valid JSON with exactly two keys: "
                    "'reasoning' and 'answer'. "
                    "'reasoning' must be at least 80 characters. "
                    "'answer' must be an integer."
                )
            },
            {
                "role": "user",
                "content": req.problem
            }
        ],
        response_format={"type": "json_object"}
    )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(repr(e))
        raise HTTPException(status_code=500, detail=str(e))