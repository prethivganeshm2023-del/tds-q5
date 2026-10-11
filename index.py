import os
import re
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]  # line numbers with errors


# ---------- Part 1: tool function ----------
def execute_python_code(code: str) -> dict:
    """Run code with exec(); return exact stdout, or the traceback on failure."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(code, {"__name__": "__main__"})
        return {"success": True, "output": sys.stdout.getvalue()}
    except Exception as e:
        # drop this function's own frame so only the user's code appears
        tb = e.__traceback__.tb_next if e.__traceback__ else None
        out = "Traceback (most recent call last):\n" if tb else ""
        out += "".join(traceback.format_tb(tb))
        out += "".join(traceback.format_exception_only(type(e), e))
        return {"success": False, "output": out}
    finally:
        sys.stdout = old_stdout


# ---------- Part 2: AI error analysis via AI Pipe (only called on failure) ----------
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
AIPIPE_MODEL = os.environ.get("AIPIPE_MODEL", "google/gemini-2.0-flash-lite-001")


def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    import httpx

    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(code.split("\n"), 1))
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) (1-indexed) in the CODE where the error occurred.

CODE (with line numbers):
{numbered}

TRACEBACK:
{tb}

Reply with JSON only, exactly in this shape: {{"error_lines": [<integers>]}}"""

    resp = httpx.post(
        AIPIPE_URL,
        headers={"Authorization": f"Bearer {os.environ['AIPIPE_TOKEN']}"},
        json={
            "model": AIPIPE_MODEL,
            "messages": [
                {"role": "system", "content": "You answer with a single JSON object and nothing else."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=25,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip()
    content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.M).strip()
    return ErrorAnalysis.model_validate_json(content).error_lines


def lines_from_traceback(tb: str) -> List[int]:
    """Safety net if the LLM call fails: last frame inside the submitted code."""
    nums = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(nums[-1])] if nums else []


@app.post("/")
@app.post("/{full_path:path}")   # answer on any path Vercel passes in (e.g. /code-interpreter, /api/index)
def code_interpreter(req: CodeRequest, full_path: str = ""):
    res = execute_python_code(req.code)
    if res["success"]:
        return {"error": [], "result": res["output"]}
    try:
        lines = analyze_error_with_ai(req.code, res["output"])
        if not lines:
            lines = lines_from_traceback(res["output"])
    except Exception:
        lines = lines_from_traceback(res["output"])
    return {"error": lines, "result": res["output"]}
