import json
import stat
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import HttpMockSequence
from mcp import Client

from syllabus_copilot.gcal import Calendar, Interval, NotSignedIn, free_slots, save_token
from syllabus_copilot.servers.google_calendar import create_server

PHX = ZoneInfo("America/Phoenix")
OK = {"status": "200"}
START = "2026-11-03T10:00:00-07:00"
END = "2026-11-03T11:00:00-07:00"
FREE = {"calendars": {"primary": {"busy": []}}}


def at(day, hour, minute=0, zone=PHX):
    return datetime(2026, 11, day, hour, minute, tzinfo=zone)


def event(id="ev1", marked=True, start=START, end=END):
    body = {
        "id": id,
        "summary": "Study: Midterm",
        "start": {"dateTime": start},
        "end": {"dateTime": end},
        "htmlLink": "https://calendar.google.com/x",
    }
    if marked:
        body["extendedProperties"] = {"private": {"syllabusCopilot": "1"}}
    return body


def harness(*responses):
    """A real Google client, checked against Google's API description, with canned responses."""
    http = HttpMockSequence(
        [(r[0], r[1]) if isinstance(r, tuple) else (OK, json.dumps(r)) for r in responses]
    )
    service = build("calendar", "v3", http=http, developerKey="test", static_discovery=True)
    return Client(create_server(lambda: Calendar(service))), http


def writes(http):
    return [
        r
        for r in http.request_sequence
        if r[1] in ("POST", "PATCH", "PUT", "DELETE") and "freeBusy" not in r[0]
    ]


def test_free_slots_skip_busy_times_and_short_gaps():
    busy = [
        Interval(at(3, 9), at(3, 10)),
        Interval(at(3, 10, 30), at(3, 12)),  # leaves a 30 minute gap, too short
        Interval(at(3, 15), at(3, 16)),
    ]

    slots = free_slots(busy, at(3, 8), at(3, 18), timedelta(hours=1), "America/Phoenix")

    assert [(s.start, s.end) for s in slots] == [
        (at(3, 8), at(3, 9)),
        (at(3, 12), at(3, 15)),
        (at(3, 16), at(3, 18)),
    ]


def test_free_slots_stay_within_waking_hours_across_days():
    slots = free_slots([], at(3, 20), at(4, 9), timedelta(hours=1), "America/Phoenix")
    assert [(s.start, s.end) for s in slots] == [(at(3, 20), at(3, 22)), (at(4, 8), at(4, 9))]


def test_free_slots_handle_overlapping_busy_times_and_other_time_zones():
    busy = [
        Interval(at(3, 9).astimezone(UTC), at(3, 11).astimezone(UTC)),
        Interval(at(3, 10), at(3, 10, 30)),
    ]
    slots = free_slots(busy, at(3, 9), at(3, 12), timedelta(hours=1), "America/Phoenix")
    assert [(s.start, s.end) for s in slots] == [(at(3, 11), at(3, 12))]


def test_free_slots_use_local_hours_across_daylight_saving():
    ny = ZoneInfo("America/New_York")
    # Clocks go back on 2026-11-01, so local 8am is a different UTC hour on each day.
    slots = free_slots(
        [],
        datetime(2026, 10, 31, 0, tzinfo=ny),
        datetime(2026, 11, 2, 0, tzinfo=ny),
        timedelta(hours=1),
        "America/New_York",
    )
    assert [s.start.hour for s in slots] == [8, 8]
    assert slots[0].start.utcoffset() != slots[1].start.utcoffset()


async def test_find_free_slots_tool_uses_google_busy_times():
    busy = {"calendars": {"primary": {"busy": [{"start": START, "end": END}]}}}
    client, http = harness(busy)
    async with client:
        result = await client.call_tool(
            "calendar_find_free_slots",
            {
                "start": "2026-11-03T09:00:00-07:00",
                "end": "2026-11-03T12:00:00-07:00",
                "duration_minutes": 60,
                "timezone": "America/Phoenix",
            },
        )
    assert not result.is_error
    ranges = result.structured_content["ranges"]
    assert [(r["start"][11:16], r["end"][11:16]) for r in ranges] == [
        ("09:00", "10:00"),
        ("11:00", "12:00"),
    ]
    assert json.loads(http.request_sequence[0][2])["items"] == [{"id": "primary"}]


async def test_calendar_read_errors_are_not_mistaken_for_free_time():
    broken = {"calendars": {"primary": {"errors": [{"reason": "notFound"}]}}}
    client, _ = harness(broken)
    async with client:
        result = await client.call_tool("calendar_get_busy", {"start": START, "end": END})
    assert result.is_error
    assert "notFound" in result.content[0].text


async def test_create_only_previews_without_confirm():
    client, http = harness(FREE)
    async with client:
        result = await client.call_tool(
            "calendar_create_study_block", {"title": "Study: Midterm", "start": START, "end": END}
        )
    assert result.structured_content["status"] == "preview"
    assert writes(http) == []


async def test_create_writes_a_marked_event_when_confirmed():
    client, http = harness(FREE, event())
    async with client:
        result = await client.call_tool(
            "calendar_create_study_block",
            {"title": "Study: Midterm", "start": START, "end": END, "confirm": True},
        )
    assert result.structured_content["status"] == "created"
    assert result.structured_content["block"]["id"] == "ev1"
    [(uri, method, body, _)] = writes(http)
    assert method == "POST" and "/calendars/primary/events" in uri
    sent = json.loads(body)
    assert sent["summary"] == "Study: Midterm"
    assert sent["extendedProperties"]["private"]["syllabusCopilot"] == "1"


async def test_preview_reports_conflicts():
    busy = {"calendars": {"primary": {"busy": [{"start": START, "end": END}]}}}
    client, _ = harness(busy)
    async with client:
        result = await client.call_tool(
            "calendar_create_study_block", {"title": "Study", "start": START, "end": END}
        )
    assert len(result.structured_content["conflicts"]) == 1


async def test_move_refuses_events_the_app_did_not_create():
    client, http = harness(event(marked=False))
    async with client:
        result = await client.call_tool(
            "calendar_move_study_block",
            {"event_id": "ev1", "new_start": START, "new_end": END, "confirm": True},
        )
    assert result.is_error
    assert "wasn't created by Syllabus Copilot" in result.content[0].text
    assert writes(http) == []


async def test_move_only_previews_without_confirm():
    client, http = harness(event(), FREE)
    async with client:
        result = await client.call_tool(
            "calendar_move_study_block",
            {
                "event_id": "ev1",
                "new_start": "2026-11-04T10:00:00-07:00",
                "new_end": "2026-11-04T11:00:00-07:00",
            },
        )
    assert result.structured_content["status"] == "preview"
    assert writes(http) == []


async def test_move_changes_the_event_when_confirmed():
    new_start, new_end = "2026-11-04T10:00:00-07:00", "2026-11-04T11:00:00-07:00"
    client, http = harness(event(), FREE, event(), event(start=new_start, end=new_end))
    async with client:
        result = await client.call_tool(
            "calendar_move_study_block",
            {"event_id": "ev1", "new_start": new_start, "new_end": new_end, "confirm": True},
        )
    assert result.structured_content["status"] == "moved"
    [(uri, method, body, _)] = writes(http)
    assert method == "PATCH" and uri.split("?")[0].endswith("/events/ev1")
    assert json.loads(body)["start"]["dateTime"] == new_start


async def test_list_only_asks_google_for_marked_events_and_follows_pages():
    client, http = harness(
        {"items": [event("a")], "nextPageToken": "next"}, {"items": [event("b")]}
    )
    async with client:
        result = await client.call_tool("calendar_list_study_blocks", {"start": START, "end": END})
    assert [b["id"] for b in result.structured_content["blocks"]] == ["a", "b"]
    assert "privateExtendedProperty=syllabusCopilot%3D1" in http.request_sequence[0][0]
    assert "pageToken=next" in http.request_sequence[1][0]


@pytest.mark.parametrize(
    "arguments",
    [
        {"start": "2026-11-03T10:00:00", "end": END},  # no UTC offset
        {"start": END, "end": START},  # ends before it starts
    ],
)
async def test_bad_times_are_rejected_before_calling_google(arguments):
    client, http = harness()
    async with client:
        result = await client.call_tool("calendar_get_busy", arguments)
    assert result.is_error
    assert http.request_sequence == []


async def test_missing_sign_in_tells_the_agent_what_to_do():
    def not_signed_in():
        raise NotSignedIn("Run `syllabus-copilot calendar-login` to sign in to Google.")

    async with Client(create_server(not_signed_in)) as client:
        result = await client.call_tool("calendar_get_busy", {"start": START, "end": END})
    assert result.is_error
    assert "calendar-login" in result.content[0].text


@pytest.mark.parametrize(
    "status, hint", [("403", "calendar-login"), ("404", "wasn't found"), ("429", "rate limit")]
)
async def test_google_errors_become_helpful_messages(status, hint):
    client, _ = harness(({"status": status}, json.dumps({"error": {"message": "nope"}})))
    async with client:
        result = await client.call_tool("calendar_get_busy", {"start": START, "end": END})
    assert result.is_error
    assert hint in result.content[0].text


def test_sign_in_token_is_private_to_the_user(tmp_path):
    path = tmp_path / "secrets" / "token.json"
    save_token(Credentials(token="t"), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
