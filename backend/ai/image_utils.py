"""
Turns an uploaded image — a filesystem path (for local testing) or raw
bytes (what a real upload endpoint will hand over) — into a base64 JPEG
string that Groq's vision API accepts.

Re-encoding everything to JPEG means we never have to guess/track the
original format or mime type, and a genuinely broken/non-image file
fails here with a clear error instead of a confusing API error later.
"""

import base64
import io

from PIL import Image, UnidentifiedImageError


class InvalidImageError(Exception):
    """Raised when the given image can't be read or decoded."""


def image_to_base64_jpeg(image) -> str:
    """
    Accepts either a filesystem path (str) or raw image bytes and
    returns a base64-encoded JPEG string.
    """
    try:
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)

        img = img.convert("RGB")  # normalizes PNG alpha / CMYK / etc.

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    except (UnidentifiedImageError, FileNotFoundError, OSError) as exc:
        raise InvalidImageError(f"Could not read image: {exc}")