"""Google Calendar access: sign-in, free/busy lookup, and study blocks.

The app only asks for free/busy times, never the titles of a person's events. The only events
it reads or changes are study blocks it created, which carry a private marker.
"""

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# calendar.events is the narrowest scope that can write to the main calendar.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
]
SECRETS_DIR = Path(os.getenv("GOOGLE_SECRETS_DIR", ".secrets"))
CLIENT_FILE = SECRETS_DIR / "google_client.json"
TOKEN_FILE = SECRETS_DIR / "google_token.json"

_MARKER = "syllabusCopilot"


class NotSignedIn(Exception):
    """There's no usable Google sign-in on this machine."""


class NotAStudyBlock(Exception):
    """The event wasn't created by this app, so it must not be changed."""


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class StudyBlock:
    id: str
    title: str
    start: datetime
    end: datetime
    link: str | None


def save_token(credentials: Credentials, path: Path = TOKEN_FILE) -> None:
    """Save the sign-in token so only this user can read it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(credentials.to_json())


def login() -> Path:
    """Sign in through the browser. Run once, from a terminal, not from an MCP server."""
    if not CLIENT_FILE.exists():
        raise NotSignedIn(f"Save your Google OAuth client file as {CLIENT_FILE} first.")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_FILE), SCOPES)
    save_token(flow.run_local_server(port=0))
    return TOKEN_FILE


def load_credentials() -> Credentials:
    if not TOKEN_FILE.exists():
        raise NotSignedIn("Run `syllabus-copilot calendar-login` to sign in to Google.")
    credentials = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if credentials.valid:
        return credentials
    try:
        credentials.refresh(Request())
    except RefreshError as error:
        raise NotSignedIn(
            "Your Google sign-in has expired. Run `syllabus-copilot calendar-login` again."
        ) from error
    save_token(credentials)
    return credentials


def _parse(value: dict) -> datetime:
    return datetime.fromisoformat(value["dateTime"])


def _to_block(event: dict) -> StudyBlock:
    return StudyBlock(
        id=event["id"],
        title=event.get("summary", ""),
        start=_parse(event["start"]),
        end=_parse(event["end"]),
        link=event.get("htmlLink"),
    )


class Calendar:
    """One Google calendar. `service` is a googleapiclient Calendar v3 client."""

    def __init__(self, service, calendar_id: str = "primary"):
        self.service = service
        self.calendar_id = calendar_id

    def busy(self, start: datetime, end: datetime) -> list[Interval]:
        response = (
            self.service.freebusy()
            .query(
                body={
                    "timeMin": start.isoformat(),
                    "timeMax": end.isoformat(),
                    "items": [{"id": self.calendar_id}],
                }
            )
            .execute()
        )
        calendar = response["calendars"][self.calendar_id]
        # Google reports some failures here with an empty busy list, which looks like free time.
        if calendar.get("errors"):
            reasons = ", ".join(e.get("reason", "unknown") for e in calendar["errors"])
            raise RuntimeError(f"Google couldn't read this calendar's busy times: {reasons}")
        return [
            Interval(datetime.fromisoformat(b["start"]), datetime.fromisoformat(b["end"]))
            for b in calendar.get("busy", [])
        ]

    def create_block(
        self, title: str, start: datetime, end: datetime, description: str | None = None
    ) -> StudyBlock:
        body = {
            "summary": title,
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
            "extendedProperties": {"private": {_MARKER: "1"}},
        }
        if description:
            body["description"] = description
        event = self.service.events().insert(calendarId=self.calendar_id, body=body).execute()
        return _to_block(event)

    def get_block(self, event_id: str) -> StudyBlock:
        event = self.service.events().get(calendarId=self.calendar_id, eventId=event_id).execute()
        if event.get("extendedProperties", {}).get("private", {}).get(_MARKER) != "1":
            raise NotAStudyBlock(event_id)
        return _to_block(event)

    def move_block(self, event_id: str, start: datetime, end: datetime) -> StudyBlock:
        self.get_block(event_id)  # Refuses anything this app didn't create.
        event = (
            self.service.events()
            .patch(
                calendarId=self.calendar_id,
                eventId=event_id,
                body={
                    "start": {"dateTime": start.isoformat()},
                    "end": {"dateTime": end.isoformat()},
                },
            )
            .execute()
        )
        return _to_block(event)

    def list_blocks(self, start: datetime, end: datetime) -> list[StudyBlock]:
        blocks: list[StudyBlock] = []
        page_token = None
        while True:
            response = (
                self.service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=start.isoformat(),
                    timeMax=end.isoformat(),
                    privateExtendedProperty=f"{_MARKER}=1",
                    singleEvents=True,
                    orderBy="startTime",
                    pageToken=page_token,
                )
                .execute()
            )
            blocks += [_to_block(event) for event in response.get("items", [])]
            page_token = response.get("nextPageToken")
            if not page_token:
                return blocks


def build_calendar() -> Calendar:
    return Calendar(build("calendar", "v3", credentials=load_credentials()))


def overlapping(busy: list[Interval], start: datetime, end: datetime) -> list[Interval]:
    return [b for b in busy if b.start < end and start < b.end]


def free_slots(
    busy: list[Interval],
    start: datetime,
    end: datetime,
    duration: timedelta,
    timezone: str,
    day_start_hour: int = 8,
    day_end_hour: int = 22,
) -> list[Interval]:
    """Free time of at least `duration` between `start` and `end`, within waking hours.

    Hours are local to `timezone`, so they stay right across daylight saving changes.
    """
    zone = ZoneInfo(timezone)
    utc = ZoneInfo("UTC")
    # Comparisons between datetimes in the same zone ignore UTC offsets, so work in UTC.
    busy_utc = sorted((b.start.astimezone(utc), b.end.astimezone(utc)) for b in busy)
    start_utc, end_utc = start.astimezone(utc), end.astimezone(utc)

    slots: list[Interval] = []

    def add(slot_start: datetime, slot_end: datetime) -> None:
        if slot_end - slot_start >= duration:
            slots.append(Interval(slot_start.astimezone(zone), slot_end.astimezone(zone)))

    day: date = start.astimezone(zone).date()
    while day <= end.astimezone(zone).date():
        window_start = datetime.combine(day, time(day_start_hour), zone).astimezone(utc)
        window_end = datetime.combine(day, time(day_end_hour), zone).astimezone(utc)
        cursor = max(window_start, start_utc)
        limit = min(window_end, end_utc)
        for busy_start, busy_end in busy_utc:
            if busy_end <= cursor or busy_start >= limit:
                continue
            add(cursor, busy_start)
            cursor = max(cursor, busy_end)
        if cursor < limit:
            add(cursor, limit)
        day += timedelta(days=1)
    return slots
