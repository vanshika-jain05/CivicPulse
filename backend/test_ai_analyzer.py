"""
Manual test for the AI complaint analyzer + full downstream pipeline.
Run from inside backend/:
    uv run python test_ai_analyzer.py
    uv run python test_ai_analyzer.py "your complaint text here"
"""

import os
import sys

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

# Drop real test photos in backend/sample_images/ — missing files are
# skipped automatically. Any customer text is ignored when an image is
# present, so these are just image paths now.
IMAGE_TEST_CASES = [
    "sample_images/pothole.jpg",
    "sample_images/garbage_pile.jpg",
    "sample_images/cat.jpg",
    "sample_images/random_selfie.jpg",
]


def process_and_print(text=None, image=None):
    print("=" * 70)
    print("Customer text:", text if text else "(none)")
    print("Image:", image if image else "(none)")

    try:
        result = analyze_complaint(complaint_text=text, image=image)
    except AIAnalysisError as e:
        print("AI analysis failed:", e)
        return

    analysis = result.analysis
    priority_score = calculate_priority(
        severity=analysis.severity,
        urgency=analysis.urgency,
        safety_risk=analysis.safety_risk,
        public_impact=analysis.public_impact,
    )
    department = get_department(analysis.category)

    print("Final stored text:", result.text)
    print("AI analysis:", analysis.model_dump())
    print("Routed to department:", department)
    print("Final priority score:", priority_score)


def run():
    for text in SAMPLE_COMPLAINTS:
        process_and_print(text=text)

    for image_path in IMAGE_TEST_CASES:
        if not os.path.exists(image_path):
            print("=" * 70)
            print(f"Skipping (no sample image found at {image_path})")
            continue
        process_and_print(image=image_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_and_print(text=" ".join(sys.argv[1:]))
    else:
        run()