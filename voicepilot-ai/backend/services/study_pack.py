"""
Generates every piece of study material for a lecture in one go.

Used by both the "Generate everything" endpoint and the worker's automatic
study-pack task, so the two can never drift apart. Only touches the ORM
object's attributes; the caller decides when to commit.
"""

import asyncio

from models.lecture import Lecture
from services.llm_service import (
    generate_chapters, generate_flashcards, generate_key_concepts, generate_mindmap,
    generate_notes, generate_quiz, generate_summary,
)


async def build_study_pack(lecture: Lecture, regenerate: bool = False) -> dict:
    """
    Fills in summary, notes, key concepts, quiz, flashcards, mind map and
    chapters. Deliberately partial-tolerant: if one generator fails (a model
    hiccup, a rate limit), the others are still applied.

    Returns {"generated": [...], "skipped": [...], "failed": {field: message}}.
    """
    segments = lecture.processed_segments or lecture.segments

    jobs = [
        ("summary", lambda: generate_summary(lecture.transcript)),
        ("notes", lambda: generate_notes(lecture.transcript)),
        ("key_concepts", lambda: generate_key_concepts(lecture.transcript)),
        ("quiz", lambda: generate_quiz(lecture.transcript, 5)),
        ("flashcards", lambda: generate_flashcards(lecture.transcript, 10)),
        ("mindmap", lambda: generate_mindmap(lecture.transcript)),
    ]
    if segments:
        jobs.append(("chapters", lambda: generate_chapters(segments)))

    generated, skipped, failed = [], [], {}

    to_run = []
    for field, generator in jobs:
        if getattr(lecture, field) and not regenerate:
            skipped.append(field)
        else:
            to_run.append((field, generator))

    # The generators are independent network calls, so run them concurrently:
    # total time ≈ the slowest one, not the sum of all. return_exceptions keeps
    # one failure from cancelling the rest. Results are applied to the ORM
    # object only after gather, so the session is never touched concurrently.
    results = await asyncio.gather(
        *(generator() for _, generator in to_run),
        return_exceptions=True,
    )

    for (field, _), result in zip(to_run, results):
        if isinstance(result, BaseException):
            failed[field] = str(result)
        else:
            setattr(lecture, field, result)
            generated.append(field)

    return {"generated": generated, "skipped": skipped, "failed": failed}
