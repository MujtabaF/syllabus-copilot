from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class DeliverableType(StrEnum):
    ASSIGNMENT = "assignment"
    QUIZ = "quiz"
    EXAM = "exam"
    PROJECT = "project"
    LAB = "lab"
    OTHER = "other"


class Deliverable(BaseModel):
    title: str
    type: DeliverableType
    due_date: date | None = Field(None, description="ISO 8601 date, or null if not stated")
    weight: float | None = Field(
        None, ge=0, le=100, description="Percent of final grade, or null if not stated"
    )


class Syllabus(BaseModel):
    course_code: str | None = None
    course_name: str | None = None
    deliverables: list[Deliverable]
