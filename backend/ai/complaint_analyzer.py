"""
AI complaint analyzer for CivicPulse — powered by Groq.

RULE:
- If an image is provided, the image is the ONLY source of truth.
- Customer text is completely ignored when an image is provided.
- The AI directly analyzes the image and generates:
    1. Complaint description
    2. Category
    3. Severity
    4. Urgency
    5. Safety risk
    6. Public impact
- If no image is provided, the customer's text is analyzed instead.
"""

import os
import json
import time
from dataclasses import dataclass

from groq import Groq, APIConnectionError, RateLimitError, APIError
from dotenv import load_dotenv
from pydantic import ValidationError

from models import ComplaintAnalysis
from ai.image_utils import image_to_base64_jpeg, InvalidImageError


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

load_dotenv()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

_FALLBACK_MODEL = os.getenv(
    "GROQ_FALLBACK_MODEL",
    "openai/gpt-oss-20b"
)

_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27b"
)

_client = Groq(api_key=_GROQ_API_KEY) if _GROQ_API_KEY else None


# ---------------------------------------------------------------------------
# VALID CATEGORIES
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RETRY CONFIGURATION
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2
_RETRY_DELAY_SECONDS = 1.5


# ---------------------------------------------------------------------------
# CUSTOM ERROR
# ---------------------------------------------------------------------------

class AIAnalysisError(Exception):
    """Raised when the AI service cannot produce a valid ComplaintAnalysis."""


# ---------------------------------------------------------------------------
# RESULT OBJECT
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    text:
        Final complaint text to store.

        If an image was analyzed:
            AI-generated description of the image.

        If no image was provided:
            Original customer text.

    analysis:
        ComplaintAnalysis containing category + four scores.
    """

    text: str
    analysis: ComplaintAnalysis


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _clamp(value, low=0.0, high=10.0):
    """
    Keep a score between 0 and 10.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return low

    return max(low, min(high, value))


def _build_analysis(raw: dict) -> ComplaintAnalysis:
    """
    Convert raw AI output into our ComplaintAnalysis model.
    """

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


# ---------------------------------------------------------------------------
# TEXT-ONLY PATH
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""
You are the complaint-triage AI for CivicPulse, a civic
issue reporting platform.

You are given the raw text a citizen typed when reporting
a problem in their city.

Read it carefully and score it.

Pick exactly ONE category from this list:

{", ".join(VALID_CATEGORIES)}

Score each from 0 to 10:

- severity: how bad the underlying problem itself is
- urgency: how quickly it needs to be acted on
- safety_risk: risk of physical harm to people if left unaddressed
- public_impact: how many people / how large an area is affected

Base every score only on what the text actually says.

If the text is vague, use moderate scores rather than
guessing extreme values.

You must respond ONLY by calling the
submit_complaint_analysis tool.
"""


_TOOL_NAME = "submit_complaint_analysis"


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Submit the structured analysis of a civic complaint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": VALID_CATEGORIES,
                    },
                    "severity": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "urgency": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "safety_risk": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "public_impact": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10,
                    },
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


def _call_groq(complaint_text: str, model: str) -> dict:
    """
    Call Groq for text-only complaint analysis.
    """

    if _client is None:
        raise AIAnalysisError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    response = _client.chat.completions.create(
        model=model,
        max_completion_tokens=300,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": complaint_text,
            },
        ],
        tools=_TOOLS,
        tool_choice={
            "type": "function",
            "function": {
                "name": _TOOL_NAME
            }
        },
    )

    message = response.choices[0].message

    for call in (message.tool_calls or []):
        if call.function.name == _TOOL_NAME:
            return json.loads(call.function.arguments)

    raise AIAnalysisError(
        f"Groq ({model}) did not return the expected tool call."
    )


def _analyze_text_only(complaint_text: str) -> ComplaintAnalysis:
    """
    Analyze complaint using text when no image exists.
    """

    models_to_try = [_MODEL]

    if _FALLBACK_MODEL and _FALLBACK_MODEL != _MODEL:
        models_to_try.append(_FALLBACK_MODEL)

    last_error = None

    for model in models_to_try:

        for attempt in range(1, _MAX_RETRIES + 2):

            try:
                raw = _call_groq(
                    complaint_text,
                    model
                )

                return _build_analysis(raw)

            except (
                APIConnectionError,
                RateLimitError,
            ) as exc:

                last_error = exc

                if attempt <= _MAX_RETRIES:
                    time.sleep(
                        _RETRY_DELAY_SECONDS * attempt
                    )
                    continue

                break

            except (
                APIError,
                ValidationError,
                json.JSONDecodeError,
                AIAnalysisError,
            ) as exc:

                last_error = exc
                break

    raise AIAnalysisError(
        f"Failed to analyze complaint: {last_error}"
    )


# ---------------------------------------------------------------------------
# IMAGE / VISION PATH
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = f"""
You are the complaint-triage AI for CivicPulse,
a civic issue reporting platform.

You are given ONE image showing a possible civic problem.

IMPORTANT RULES:

1. The IMAGE is the ONLY source of truth.
2. Do NOT use customer text.
3. Do NOT invent facts that cannot be seen.
4. Analyze only what is visibly present.
5. Generate the complaint description directly from the image.
6. Generate the category directly from the image.
7. Generate all four scores directly from the image.

Pick exactly ONE category:

{", ".join(VALID_CATEGORIES)}

Categories:

- road_damage = potholes, damaged roads, broken pavement
- garbage = garbage, waste, overflowing bins
- street_light = broken/non-working street lights
- naked_wires = exposed or hanging electrical wires
- power_outage = visible evidence related to power outage
- water = leakage, flooding, water-related civic problems
- crime = visible evidence of a crime/security issue
- fire = visible fire or fire-related damage
- other = anything that does not clearly fit above

Score each from 0 to 10.

severity:
How serious is the visible problem itself?

urgency:
How quickly should authorities respond?

safety_risk:
How much physical danger does the visible problem pose?

public_impact:
How many people or how large an area appears affected?

IMPORTANT:
Do not give extreme scores unless the image clearly supports them.

If the image does NOT clearly show a civic problem:
- category = "other"
- use moderate/low scores
- describe what is actually visible

Return a SINGLE valid JSON object.

The JSON MUST contain exactly these six keys:

{{
    "complaint_description": "short factual description",
    "category": "one valid category",
    "severity": 0,
    "urgency": 0,
    "safety_risk": 0,
    "public_impact": 0
}}

Return JSON only.
No markdown.
No ```json.
No explanation.
No reasoning.
"""


def _call_groq_vision(image_b64: str) -> dict:
    """
    Analyze an image directly using Groq's vision model.

    The model itself generates:
        complaint_description
        category
        severity
        urgency
        safety_risk
        public_impact
    """

    if _client is None:
        raise AIAnalysisError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    response = _client.chat.completions.create(
        model=_VISION_MODEL,

        # Enough room for the JSON response.
        max_completion_tokens=500,

        # Lower temperature = more consistent scoring.
        temperature=0.2,

        # IMPORTANT:
        # Qwen 3.6 supports JSON mode with image inputs.
        response_format={
            "type": "json_object"
        },

        # Do not return thinking/reasoning in the content.
        reasoning_format="hidden",

        # We do not need reasoning for this classification task.
        reasoning_effort="none",

        messages=[
            {
                "role": "system",
                "content": _VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this image carefully. "
                            "Analyze the visible civic issue and "
                            "return ONLY the required JSON object."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                f"{image_b64}"
                            )
                        },
                    },
                ],
            },
        ],
    )

    message = response.choices[0].message

    content = message.content

    if not content or not content.strip():

        # This gives us a much more useful error if the model
        # returns no visible answer.
        raise AIAnalysisError(
            f"Groq vision model ({_VISION_MODEL}) "
            "returned an empty response."
        )

    try:
        raw = json.loads(content)

    except json.JSONDecodeError as exc:

        raise AIAnalysisError(
            f"Groq vision model ({_VISION_MODEL}) "
            f"returned invalid JSON: {exc}. "
            f"Raw response: {content[:500]}"
        )

    if not isinstance(raw, dict):
        raise AIAnalysisError(
            "Groq vision model returned JSON, "
            "but it was not a JSON object."
        )

    return raw


def _analyze_with_image(image_b64: str) -> AnalysisResult:
    """
    Image is the sole source of truth.

    The vision model directly generates:
        description + category + 4 scores.
    """

    last_error = None

    for attempt in range(1, _MAX_RETRIES + 2):

        try:

            raw = _call_groq_vision(image_b64)

            extracted_text = (
                raw.get("complaint_description") or ""
            ).strip()

            if not extracted_text:
                extracted_text = (
                    "Complaint filed from an uploaded image."
                )

            analysis = _build_analysis(raw)

            return AnalysisResult(
                text=extracted_text,
                analysis=analysis,
            )

        except (
            APIConnectionError,
            RateLimitError,
        ) as exc:

            last_error = exc

            if attempt <= _MAX_RETRIES:
                time.sleep(
                    _RETRY_DELAY_SECONDS * attempt
                )
                continue

            break

        except (
            APIError,
            ValidationError,
            json.JSONDecodeError,
            AIAnalysisError,
        ) as exc:

            last_error = exc

            # Retry once/twice for model-generation issues.
            if attempt <= _MAX_RETRIES:
                time.sleep(
                    _RETRY_DELAY_SECONDS * attempt
                )
                continue

            break

    raise AIAnalysisError(
        f"Failed to analyze complaint image: {last_error}"
    )


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def analyze_complaint(
    complaint_text: str = None,
    image=None
) -> AnalysisResult:
    """
    Analyze a civic complaint.

    PRIORITY:

    1. If image exists:
           image is the ONLY source of truth.
           complaint_text is ignored.

    2. If no image exists:
           complaint_text is analyzed.

    image may be:
        - filesystem path
        - raw image bytes
    """

    has_text = bool(
        complaint_text and complaint_text.strip()
    )

    has_image = image is not None

    if not has_text and not has_image:
        raise AIAnalysisError(
            "Provide complaint_text, an image, or both."
        )

    # -------------------------------------------------------
    # IMAGE PATH
    # -------------------------------------------------------

    if has_image:

        try:
            image_b64 = image_to_base64_jpeg(image)

        except InvalidImageError as exc:
            raise AIAnalysisError(str(exc))

        # IMPORTANT:
        # Do NOT pass complaint_text here.
        # Image is the only source of truth.
        return _analyze_with_image(image_b64)

    # -------------------------------------------------------
    # TEXT PATH
    # -------------------------------------------------------

    analysis = _analyze_text_only(
        complaint_text
    )

    return AnalysisResult(
        text=complaint_text,
        analysis=analysis,
    )