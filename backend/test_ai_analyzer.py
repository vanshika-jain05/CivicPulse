"""
Manual test for the AI complaint analyzer + full downstream pipeline.
Run from inside backend/:
    uv run python test_ai_analyzer.py
"""

from ai.complaint_analyzer import analyze_complaint, AIAnalysisError
from priority import calculate_priority, get_department

SAMPLE_COMPLAINTS = [
    "There is a huge pothole on MG Road near the bus stop, two bikers "
    "have already fallen and hurt themselves this week.",

    "Garbage has not been collected from our street in Sector 12 for "
    "5 days, it is starting to smell very bad.",

    "The streetlight outside house number 45 in my colony has been off "
    "for two weeks, it's very dark and unsafe to walk at night.",

    "There are live wires hanging low near the park entrance, children "
    "play there every evening.",

    "No power in our entire block since this morning, it's exam day "
    "for my kids and they can't study.",
]


def run():
    for text in SAMPLE_COMPLAINTS:
        print("=" * 70)
        print("Complaint:", text)

        try:
            analysis = analyze_complaint(text)

        except AIAnalysisError as e:
            print("AI analysis failed:", e)
            continue

        priority_score = calculate_priority(
            severity=analysis.severity,
            urgency=analysis.urgency,
            safety_risk=analysis.safety_risk,
            public_impact=analysis.public_impact,
        )
        department = get_department(analysis.category)

        print("AI analysis:", analysis.model_dump())
        print("Routed to department:", department)
        print("Final priority score:", priority_score)


import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run a single custom complaint: uv run python test_ai_analyzer.py "your text here"
        custom_text = " ".join(sys.argv[1:])
        SAMPLE_COMPLAINTS.append(custom_text)

    run()