"""MCP server: find free time in Google Calendar and manage study blocks.

Tools that write only preview the change unless called with `confirm=true`, so a person can
approve it first.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Annotated, Literal
from zoneinfo import ZoneInfoNotFoundError

from google.auth.exceptions import GoogleAuthError
from googleapiclient.errors import HttpError
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import AwareDatetime, BaseModel, Field

from ..gcal import (
    Calendar,
    Interval,
    NotAStudyBlock,
    NotSignedIn,
    StudyBlock,
    build_calendar,
    free_slots,
    overlapping,
)


class TimeRange(BaseModel):
    start: datetime
    end: datetime


class TimeRanges(BaseModel):
    ranges: list[TimeRange]


class Block(BaseModel):
    id: str
    title: str
    start: datetime
    end: datetime
    link: str | None


class Blocks(BaseModel):
    blocks: list[Block]


class CreateResult(BaseModel):
    status: Literal["preview", "created"]
    title: str
    start: datetime
    end: datetime
    conflicts: list[TimeRange] = Field(description="Busy times this block would overlap")
    block: Block | None = None


class MoveResult(BaseModel):
    status: Literal["preview", "moved"]
    block: Block
    new_start: datetime
    new_end: datetime
    conflicts: list[TimeRange] = Field(description="Busy times the new time would overlap")


def _range(interval: Interval) -> TimeRange:
    return TimeRange(start=interval.start, end=interval.end)


def _block(block: StudyBlock) -> Block:
    return Block(id=block.id, title=block.title, start=block.start, end=block.end, link=block.link)


def _check_order(start: datetime, end: datetime) -> None:
    if end <= start:
        raise ToolError("The end time must be after the start time.")


@contextmanager
def _friendly_errors() -> Iterator[None]:
    """Turn expected failures into messages that tell the agent what to do next."""
    try:
        yield
    except NotSignedIn as error:
        raise ToolError(str(error)) from error
    except NotAStudyBlock as error:
        raise ToolError(
            f"Event {error} wasn't created by Syllabus Copilot, so it can't be changed here."
        ) from error
    except GoogleAuthError as error:
        raise ToolError(
            "Google sign-in failed. Run `syllabus-copilot calendar-login` again."
        ) from error
    except ZoneInfoNotFoundError as error:
        raise ToolError(f"Unknown time zone. Use a name like America/Phoenix. ({error})") from error
    except HttpError as error:
        status = error.resp.status
        if status in (401, 403):
            hint = "Run `syllabus-copilot calendar-login` again, or check calendar permissions."
        elif status == 404:
            hint = "That event wasn't found. Use calendar_list_study_blocks to get a valid id."
        elif status == 429:
            hint = "Google's rate limit was hit. Try again in a minute."
        else:
            hint = error.reason
        raise ToolError(f"Google Calendar returned {status}: {hint}") from error
    except RuntimeError as error:  # Calendar.busy reports per-calendar errors this way.
        raise ToolError(str(error)) from error


def create_server(get_calendar: Callable[[], Calendar] = build_calendar) -> MCPServer:
    """`get_calendar` runs on each call, so the server can start before anyone has signed in."""
    server = MCPServer(
        "calendar",
        instructions=(
            "Find free time and manage study blocks in the user's Google Calendar. Show the user "
            "each write's preview and get their approval before calling it with confirm=true."
        ),
    )

    @server.tool(
        name="calendar_get_busy",
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    )
    def get_busy(start: AwareDatetime, end: AwareDatetime) -> TimeRanges:
        """List when the user is busy, without any event details.

        Times need a UTC offset, e.g. 2026-11-03T09:00:00-07:00.
        """
        _check_order(start, end)
        with _friendly_errors():
            return TimeRanges(ranges=[_range(b) for b in get_calendar().busy(start, end)])

    @server.tool(
        name="calendar_find_free_slots",
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    )
    def find_free_slots(
        start: AwareDatetime,
        end: AwareDatetime,
        duration_minutes: int = Field(ge=5, le=720),
        timezone: str = Field(description="IANA time zone name, e.g. America/Phoenix"),
        day_start_hour: int = Field(default=8, ge=0, le=23),
        day_end_hour: int = Field(default=22, ge=1, le=23),
    ) -> TimeRanges:
        """Find free time of at least `duration_minutes`, within the given hours of each day.

        Times need a UTC offset, e.g. 2026-11-03T09:00:00-07:00.
        """
        _check_order(start, end)
        if day_end_hour <= day_start_hour:
            raise ToolError("day_end_hour must be after day_start_hour.")
        with _friendly_errors():
            slots = free_slots(
                get_calendar().busy(start, end),
                start,
                end,
                timedelta(minutes=duration_minutes),
                timezone,
                day_start_hour,
                day_end_hour,
            )
            return TimeRanges(ranges=[_range(s) for s in slots])

    @server.tool(
        name="calendar_list_study_blocks",
        annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    )
    def list_study_blocks(start: AwareDatetime, end: AwareDatetime) -> Blocks:
        """List study blocks created by Syllabus Copilot. Other events are never included."""
        _check_order(start, end)
        with _friendly_errors():
            return Blocks(blocks=[_block(b) for b in get_calendar().list_blocks(start, end)])

    @server.tool(
        name="calendar_create_study_block",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False),
    )
    def create_study_block(
        title: Annotated[str, Field(min_length=1, max_length=200)],
        start: AwareDatetime,
        end: AwareDatetime,
        description: str | None = None,
        confirm: bool = False,
    ) -> CreateResult:
        """Add a study block. Only previews it, including any conflicts, unless `confirm` is true.

        Get the user's approval of the preview before confirming.
        """
        _check_order(start, end)
        with _friendly_errors():
            calendar = get_calendar()
            conflicts = overlapping(calendar.busy(start, end), start, end)
            block = calendar.create_block(title, start, end, description) if confirm else None
        return CreateResult(
            status="created" if confirm else "preview",
            title=title,
            start=start,
            end=end,
            conflicts=[_range(c) for c in conflicts],
            block=_block(block) if block else None,
        )

    @server.tool(
        name="calendar_move_study_block",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True
        ),
    )
    def move_study_block(
        event_id: str,
        new_start: AwareDatetime,
        new_end: AwareDatetime,
        confirm: bool = False,
    ) -> MoveResult:
        """Move a study block. Only previews the move unless `confirm` is true.

        Works only on blocks created by Syllabus Copilot. Get the user's approval first.
        """
        _check_order(new_start, new_end)
        with _friendly_errors():
            calendar = get_calendar()
            current = calendar.get_block(event_id)
            conflicts = [
                b
                for b in overlapping(calendar.busy(new_start, new_end), new_start, new_end)
                if not (b.start == current.start and b.end == current.end)
            ]
            block = calendar.move_block(event_id, new_start, new_end) if confirm else current
        return MoveResult(
            status="moved" if confirm else "preview",
            block=_block(block),
            new_start=new_start,
            new_end=new_end,
            conflicts=[_range(c) for c in conflicts],
        )

    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
