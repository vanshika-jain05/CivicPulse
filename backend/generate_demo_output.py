# backend/generate_demo_output.py
"""
Run this once (with a working GROQ_API_KEY) to capture real AI output
into a markdown file you commit to the repo — so judges/reviewers can
see proof it works even if the live API is down or rate-limited later.

Run:
    uv run python generate_demo_output.py
"""

from ai.complaint_analyzer import analyze_complaint, AIAnalysisError
from priority import calculate_priority, get_department
from test_ai_analyzer import SAMPLE_COMPLAINTS

OUTPUT_FILE = "demo_output.md"


def run():
    lines = ["# CivicPulse AI — Sample Output\n",
             "Generated once from real Groq API calls, committed as proof of working integration.\n"]

    for text in SAMPLE_COMPLAINTS:
        lines.append(f"### Complaint\n> {text}\n")
        try:
            analysis = analyze_complaint(text)
        except AIAnalysisError as e:
            lines.append(f"**AI call failed:** {e}\n")
            continue

        priority_score = calculate_priority(
            severity=analysis.severity,
            urgency=analysis.urgency,
            safety_risk=analysis.safety_risk,
            public_impact=analysis.public_impact,
        )
        department = get_department(analysis.category)

        lines.append(f"- **Category:** {analysis.category}")
        lines.append(f"- **Severity:** {analysis.severity}")
        lines.append(f"- **Urgency:** {analysis.urgency}")
        lines.append(f"- **Safety risk:** {analysis.safety_risk}")
        lines.append(f"- **Public impact:** {analysis.public_impact}")
        lines.append(f"- **Priority score:** {priority_score}")
        lines.append(f"- **Routed to:** {department}\n")

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    run()