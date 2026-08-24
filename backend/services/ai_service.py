"""
Bridges the AI component to the rest of the backend.

main.py calls analyze(text) and wraps the result in ComplaintAnalysis(),
so this function must keep returning a dict with exactly those keys.
The actual AI logic (Groq call, prompt, validation, retries) lives in
ai/complaint_analyzer.py — this file just connects the two.
"""

from ai.complaint_analyzer import analyze_complaint, AIAnalysisError


def analyze(text: str):
    try:
        analysis = analyze_complaint(text)
        return analysis.model_dump()

    except AIAnalysisError as e:
        # Fallback so a single AI hiccup doesn't take down complaint creation.
        # Mid-range, non-committal scores + "other" category route it to
        # Municipal Administration for manual review instead of crashing.
        print(f"[ai_service] AI analysis failed, using fallback: {e}")

        return {
            "category": "other",
            "severity": 5,
            "urgency": 5,
            "safety_risk": 5,
            "public_impact": 5,
        }