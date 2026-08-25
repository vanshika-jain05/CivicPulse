"""
Run this once (with a working GROQ_API_KEY) to capture real AI output —
text and image cases — into a markdown file you commit as proof the
integration works.

Run:
    uv run python generate_demo_output.py
"""

import os

from ai.complaint_analyzer import analyze_complaint, AIAnalysisError
from priority import calculate_priority, get_department
from test_ai_analyzer import SAMPLE_COMPLAINTS, IMAGE_TEST_CASES

OUTPUT_FILE = "demo_output.md"


def analyze_and_format(text=None, image=None):
    lines = ["### Complaint"]
    if text:
        lines.append(f"> Customer text: {text}\n")
    if image:
        lines.append(f"> Image: `{image}`\n")

    try:
        result = analyze_complaint(complaint_text=text, image=image)
    except AIAnalysisError as e:
        lines.append(f"**AI call failed:** {e}\n")
        return lines

    analysis = result.analysis
    priority_score = calculate_priority(
        severity=analysis.severity,
        urgency=analysis.urgency,
        safety_risk=analysis.safety_risk,
        public_impact=analysis.public_impact,
    )
    department = get_department(analysis.category)

    lines.append(f"- **Final stored text:** {result.text}")
    lines.append(f"- **Category:** {analysis.category}")
    lines.append(f"- **Severity:** {analysis.severity}")
    lines.append(f"- **Urgency:** {analysis.urgency}")
    lines.append(f"- **Safety risk:** {analysis.safety_risk}")
    lines.append(f"- **Public impact:** {analysis.public_impact}")
    lines.append(f"- **Priority score:** {priority_score}")
    lines.append(f"- **Routed to:** {department}\n")
    return lines


def run():
    lines = [
        "# CivicPulse AI — Sample Output\n",
        "Generated from real Groq API calls, committed as proof of working integration.\n",
        "## Text-only complaints\n",
    ]

    for text in SAMPLE_COMPLAINTS:
        lines.extend(analyze_and_format(text=text))

    image_cases = [p for p in IMAGE_TEST_CASES if os.path.exists(p)]
    if image_cases:
        lines.append("## Image-based complaints (text extracted from image)\n")
        for image_path in image_cases:
            lines.extend(analyze_and_format(image=image_path))

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    run()