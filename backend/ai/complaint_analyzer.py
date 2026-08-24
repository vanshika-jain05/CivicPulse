"""
Given the raw text of a citizen complaint, this asks a Groq-hosted model
to score it along four axes and pick a category, then returns the
project's existing ComplaintAnalysis model (from models.py). It does NOT
calculate priority or department — priority.py still owns that.
"""

import os
import json
import time

from groq import Groq, APIConnectionError, RateLimitError, APIError
from dotenv import load_dotenv
from pydantic import ValidationError

from models import ComplaintAnalysis

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None

# Must exactly match the categories priority.get_department() understands.
# "other" is a safe catch-all so unmatched complaints still get a sane
# fallback department instead of crashing anything downstream.
VALID_CATEGORIES = [
    "road_damage",
    "garbage",
    "street_light",
    "naked_wires",
    "power_outage",
    "water",
    "crime",
    "fire",
    "other",
]

_SYSTEM_PROMPT = f"""You are the complaint-triage AI for CivicPulse, a civic
issue reporting platform. You are given the raw text a citizen typed when
reporting a problem in their city. Read it carefully and score it.

Pick exactly one category from this list (use "other" only if truly none fit):
{", ".join(VALID_CATEGORIES)}

Score each of these from 0 (none) to 10 (extreme):
- severity: how bad the underlying problem itself is
- urgency: how quickly it needs to be acted on
- safety_risk: risk of physical harm to people if left unaddressed
- public_impact: how many people / how large an area is affected

Base every score only on what the text actually says. If the text is
vague, use moderate (mid-range) scores rather than guessing extremes.

You must respond ONLY by calling the submit_complaint_analysis tool."""

_TOOL_NAME = "submit_complaint_analysis"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": "Submit the structured analysis of a civic complaint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": VALID_CATEGORIES,
                        "description": "Best-fitting complaint category.",
                    },
                    "severity": {"type": "number", "minimum": 0, "maximum": 10},
                    "urgency": {"type": "number", "minimum": 0, "maximum": 10},
                    "safety_risk": {"type": "number", "minimum": 0, "maximum": 10},
                    "public_impact": {"type": "number", "minimum": 0, "maximum": 10},
                },
                "required": [
                    "category",
                    "severity",
                    "urgency",
                    "safety_risk",
                    "public_impact",
                ],
            },
        },
    }
]

_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 1.5


class AIAnalysisError(Exception):
    """Raised when the AI service cannot produce a valid ComplaintAnalysis."""


def _clamp(value, low=0.0, high=10.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def _call_groq(complaint_text: str) -> dict:
    if _client is None:
        raise AIAnalysisError("GROQ_API_KEY is not set. Add it to your .env file.")

    response = _client.chat.completions.create(
        model=_MODEL,
        max_completion_tokens=300,
        temperature=0.2,  # low temperature = more consistent scoring
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": complaint_text},
        ],
        tools=_TOOLS,
        tool_choice={"type": "function", "function": {"name": _TOOL_NAME}},
    )

    message = response.choices[0].message
    tool_calls = message.tool_calls or []

    for call in tool_calls:
        if call.function.name == _TOOL_NAME:
            return json.loads(call.function.arguments)

    raise AIAnalysisError("Groq did not return the expected tool call.")


def analyze_complaint(complaint_text: str) -> ComplaintAnalysis:
    """
    Analyze a citizen complaint's text and return a validated ComplaintAnalysis.
    Raises AIAnalysisError if the AI call or validation fails after retries.
    """
    if not complaint_text or not complaint_text.strip():
        raise AIAnalysisError("complaint_text is empty.")

    last_error = None

    for attempt in range(1, _MAX_RETRIES + 2):  # 1 initial try + retries
        try:
            raw = _call_groq(complaint_text)

            category = raw.get("category", "other")
            if category not in VALID_CATEGORIES:
                category = "other"

            cleaned = {
                "category": category,
                "severity": _clamp(raw.get("severity")),
                "urgency": _clamp(raw.get("urgency")),
                "safety_risk": _clamp(raw.get("safety_risk")),
                "public_impact": _clamp(raw.get("public_impact")),
            }

            return ComplaintAnalysis(**cleaned)

        except (APIConnectionError, RateLimitError) as exc:
            last_error = exc
            if attempt <= _MAX_RETRIES:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
                continue
            break

        except (APIError, ValidationError, json.JSONDecodeError, AIAnalysisError) as exc:
            last_error = exc
            break

    raise AIAnalysisError(f"Failed to analyze complaint: {last_error}")