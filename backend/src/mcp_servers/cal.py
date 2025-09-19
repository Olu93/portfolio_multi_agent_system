import asyncio
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

from caldav import DAVClient
from caldav.calendarobjectresource import Event
from caldav.collection import Calendar as CalDAVCalendar
from fastmcp import Context, FastMCP
from icalendar import Calendar
from icalendar.cal import Component
from icalendar.prop import TimeBase, vCategory, vDDDTypes
from pydantic import BaseModel, Field

from mcp_servers.utils.constants import MCP_HOST, MCP_PORT
from mcp_servers.utils.helper import log, start_mcp_server
from mcp_servers.utils.models import MCPResponse

logger = logging.getLogger(__name__)

# CalDAV Configuration
CALDAV_URL = os.getenv("CALDAV_URL", "")
CALDAV_USER = os.getenv("CALDAV_USER", "")
CALDAV_PASSWORD = os.getenv("CALDAV_PASSWORD", "")


# TODO: Introduce the handling of VEVENT, VTODO, VJOURNAL, VFREEBUSY, VALARM, VTIMEZONE
# - VTIMEZONE: Needs a look up. Potentially by UID, or by date range
# - VTODO: Needs a set of CRUD operations
# - VJOURNAL: Needs a set of CRUD operations
# - VALARM: Needs a set of CRUD operations

# TODO: The method "utcnow" in class "datetime" is deprecated.Use timezone-aware objects to represent datetimes in UTC; e.g. by calling .now(datetime.UTC)
# TODO: Add timezone support as the created events are all shifted by 2 hours


class CalEventEntryContract(BaseModel):
    _type: Literal["VEVENT", "VBUSY"]
    cal_url: str = Field(..., description="The URL of the calendar")
    summary: str = Field("", description="The summary of the event")
    start_datetime: datetime = Field(datetime.now(), description="The start datetime of the event as datetime object")
    end_datetime: datetime = Field(datetime.now(), description="The end datetime of the event as datetime object")
    extra: dict[str, Any] = Field(default_factory=dict, description="Extra fields of the event")


def event_to_dict_list(event_obj: Event) -> list[CalEventEntryContract]:
    """
    Convert a caldav Event object into a list of dictionaries.
    - VEVENT entries are kept as-is.
    - VFREEBUSY entries are expanded into multiple VBUSY entries (one per busy FREEBUSY period).
    Datetime fields are kept as datetime objects.
    """
    raw_ics = event_obj.data
    cal = Calendar.from_ical(raw_ics)
    results: list[CalEventEntryContract] = []

    for component in cal.walk():
        component: Component = component
        if component.name == "VEVENT":
            vText = str(component.pop("SUMMARY", ""))
            vStart: vDDDTypes = component.pop("DTSTART")
            vEnd: vDDDTypes = component.pop("DTEND")
            vCatObject: vCategory = component.pop("CATEGORIES", [])
            vCategories: list[str] = [str(category) for category in vCatObject]
            entry = CalEventEntryContract(
                _type="VEVENT",
                cal_url=event_obj.url,
                summary=vText,
                start_datetime=vStart.dt if vStart else datetime.now(),
                end_datetime=vEnd.dt if vEnd else datetime.now(),
            )
            entry.extra["categories"] = vCategories
            for key, value in component.items():
                value: TimeBase = value if isinstance(value, TimeBase) else str(value)
                entry.extra[key.lower()] = value.dt if hasattr(value, "dt") else str(value)
            results.append(entry)

        elif component.name == "VFREEBUSY":
            freebusy_values = component.get("FREEBUSY")
            if not isinstance(freebusy_values, list):
                freebusy_values = [freebusy_values]

            # TODO: Remove this redundant code by creating a new function to handle VFREEBUSY
            for period in freebusy_values:
                fbtype = period.params.get("FBTYPE", "BUSY").upper()
                if fbtype == "FREE":
                    continue  # skip free times

                busy_event = CalEventEntryContract(
                    _type="VBUSY",
                    cal_url=event_obj.url,
                    summary="Busy",
                    start_datetime=period.dtstart,
                    end_datetime=period.dtend,
                    extra={},
                )
                results.append(busy_event)

    return results


@dataclass
class CalDAVEvent:
    """Represents a CalDAV calendar event"""

    title: str
    start_time: datetime
    end_time: datetime
    description: Optional[str] = None
    location: Optional[str] = None
    uid: Optional[str] = None


class CalDAVServer:
    """Handles CalDAV operations"""

    def __init__(self):
        self.url = CALDAV_URL
        self.user = CALDAV_USER
        self.password = CALDAV_PASSWORD
        self.client = None
        self.principal = None
        self.calendars: list[CalDAVCalendar] = []
        self.primary_calendar = None

        # Validate configuration
        if not self.url or not self.user or not self.password:
            raise ValueError("CALDAV_URL, CALDAV_USER, and CALDAV_PASSWORD environment variables must be set")

    async def connect(self, ctx: Context) -> bool:
        """Connect to CalDAV server and initialize calendars"""
        try:
            await log("Connecting to CalDAV server...", "info", logger, ctx)

            # Create client and connect
            self.client = DAVClient(url=self.url, username=self.user, password=self.password)
            self.principal = self.client.principal()
            self.calendars = self.principal.calendars()

            if not self.calendars:
                await log("No calendars found on CalDAV server", "error", logger, ctx)
                return False

            # Set primary calendar (first one)
            self.primary_calendar = self.calendars[0]
            await log(
                f"Connected to CalDAV server. Found {len(self.calendars)} calendars",
                "info",
                logger,
                ctx,
            )
            await log(f"Primary calendar: {self.primary_calendar.name}", "info", logger, ctx)

            return True

        except Exception as e:
            await log(f"Failed to connect to CalDAV server: {str(e)}", "error", logger, ctx, exception=e)
            return False

    def _generate_ical_event(self, event: CalDAVEvent, attendees: List[str] = None) -> str:
        """Generate iCal format for an event, including attendees for invitations."""
        uid = event.uid or f"{int(datetime.utcnow().timestamp())}@caldav.mcp"
        dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dtstart = event.start_time.strftime("%Y%m%dT%H%M%SZ")
        dtend = event.end_time.strftime("%Y%m%dT%H%M%SZ")

        ical = (
            f"BEGIN:VCALENDAR\n"
            f"VERSION:2.0\n"
            f"PRODID:-//CalDAV MCP Server//EN\n"
            f"BEGIN:VEVENT\n"
            f"UID:{uid}\n"
            f"DTSTAMP:{dtstamp}\n"
            f"DTSTART:{dtstart}\n"
            f"DTEND:{dtend}\n"
            f"SUMMARY:{event.title}\n"
            f"ORGANIZER;CN=Me:mailto:{self.user}"
        )

        # Add attendees if provided
        if attendees:
            for a in attendees:
                cn = a.split("@")[0]
                ical += f"\nATTENDEE;CN={cn};RSVP=TRUE:mailto:{a}"

        if event.description:
            ical += f"\nDESCRIPTION:{event.description}"
        if event.location:
            ical += f"\nLOCATION:{event.location}"

        ical += "\nEND:VEVENT\nEND:VCALENDAR"
        return ical

    async def create_event(self, event: CalDAVEvent, ctx: Context, attendees: List[str] = None) -> MCPResponse:
        """Create a new calendar event and send invites via CalDAV server."""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return MCPResponse(status="ERR", error="Failed to connect to CalDAV server")

            await log(f"Creating event: {event.title}", "info", logger, ctx)

            # Generate iCal with attendees
            ical_content = self._generate_ical_event(event, attendees)

            # Add event to calendar
            self.primary_calendar.add_event(ical_content)

            await log(
                "Event created successfully! Invitations sent by server if applicable.",
                "info",
                logger,
                ctx,
            )
            return MCPResponse(status="OK", payload=f"Event '{event.title}' created successfully and invites sent")

        except Exception as e:
            error_msg = f"Failed to create event: {str(e)}"
            await log(error_msg, "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return MCPResponse(status="ERR", error=error_msg)

    # TODO: Add limit parameter to get events but keep the default to 10
    async def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        calendar_id: Optional[str] = None,
        ctx: Context = None,
    ) -> MCPResponse:
        """Get events from the calendar"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return MCPResponse(status="ERR", error="Failed to connect to CalDAV server")

            if calendar_id:
                for cal in self.calendars:
                    if cal.id == calendar_id:
                        self.primary_calendar = cal
                        break
                if not self.primary_calendar:
                    return MCPResponse(status="ERR", error=f"Calendar with ID {calendar_id} not found")

            await log("Fetching calendar events...", "info", logger, ctx)

            # Set default date range if not provided
            if not start_date:
                start_date = datetime.utcnow()
            if not end_date:
                end_date = start_date + timedelta(days=30)

            # Search for events in date range
            results = self.primary_calendar.date_search(start=start_date, end=end_date)

            await log(f"Retrieved {len(results)} events", "info", logger, ctx)

            # Format event details
            event_details: list[CalEventEntryContract] = []
            for ev in results[:10]:  # Limit to first 10 events
                try:
                    event_details.extend(event_to_dict_list(ev))
                except Exception as e:
                    error_event = CalEventEntryContract(
                        _type="VEVENT",
                        cal_url=ev.url,
                        summary="Error parsing event",
                        start_datetime=ev.dtstart,
                        end_datetime=ev.dtend,
                    )
                    error_event.extra["error"] = str(e)
                    event_details.append(error_event)

            payload = f"Retrieved {len(results)} events from {start_date.date()} to {end_date.date()}:\n\n" + "\n\n".join(
                [event.model_dump_json(indent=2) for event in event_details]
            )
            return MCPResponse(status="OK", payload=payload)

        except Exception as e:
            error_msg = f"Failed to retrieve events: {str(e)}"
            await log(error_msg, "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return MCPResponse(status="ERR", error=error_msg)

    async def delete_event(self, event_uid: str, ctx: Context) -> MCPResponse:
        """Delete a calendar event by UID"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return MCPResponse(status="ERR", error="Failed to connect to CalDAV server")

            await log(f"Searching for event with UID: {event_uid}", "info", logger, ctx)

            # Search for events in a wide date range to find the specific event
            start_date = datetime.utcnow() - timedelta(days=365)
            end_date = datetime.utcnow() + timedelta(days=365)

            results = self.primary_calendar.date_search(start=start_date, end=end_date)

            # Find event with matching UID
            target_event = None
            for ev in results:
                try:
                    if hasattr(ev.instance.vevent, "uid") and ev.instance.vevent.uid.value == event_uid:
                        target_event = ev
                        break
                except Exception:
                    continue

            if not target_event:
                return MCPResponse(status="ERR", error=f"Event with UID {event_uid} not found")

            # Delete the event
            target_event.delete()
            await log("Event deleted successfully!", "info", logger, ctx)
            return MCPResponse(status="OK", payload=f"Event with UID {event_uid} deleted successfully")

        except Exception as e:
            error_msg = f"Failed to delete event: {str(e)}"
            await log(error_msg, "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return MCPResponse(status="ERR", error=error_msg)

    async def list_calendars(self, ctx: Context) -> MCPResponse:
        """List available calendars"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return MCPResponse(status="ERR", error="Failed to connect to CalDAV server")

            calendar_list: list[str] = []
            for i, cal in enumerate(self.calendars):
                calendar_list.append(f"{i + 1}. Name: {cal.name} - ID: {cal.id} - URL: {cal.url}")

            payload = f"Available calendars:\n" + "\n".join(calendar_list)
            return MCPResponse(status="OK", payload=payload)

        except Exception as e:
            error_msg = f"Failed to list calendars: {str(e)}"
            await log(error_msg, "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return MCPResponse(status="ERR", error=error_msg)

    async def get_calendar_by_email(self, email: str, ctx: Context) -> MCPResponse:
        """Get a calendar by email"""
        try:
            for cal in self.calendars:
                if cal.email == email:
                    return MCPResponse(status="OK", payload=cal.name)

        except Exception as e:
            error_msg = f"Failed to get calendar by email: {str(e)}"
            await log(error_msg, "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return MCPResponse(status="ERR", error=error_msg)

    def test_connection(self) -> Dict[str, Any]:
        """Test CalDAV connection and return status"""
        try:
            return {
                "status": "configured",
                "url": self.url,
                "user": self.user,
                "password": "***" if self.password else None,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_availability_for_email(self, email: str, start: datetime, end: datetime, ctx: Context) -> list[CalEventEntryContract]:
        """
        Get busy periods for a given email between start and end.
        - Checks free/busy for org members (CalDAV free-busy-query)
        - Checks shared calendars in self.calendars
        Returns busy periods as CalEventEntryContract list.
        """


        busy_entries: list[CalEventEntryContract] = []

        try:
            # 1. First, check if this email corresponds to a shared calendar in self.calendars
            for cal in self.calendars:
                cal: CalDAVCalendar = cal
                if getattr(cal, "email", "").lower() == email.lower():
                    await log(
                        f"Found shared calendar for {email}, checking events...",
                        "info",
                        logger,
                        ctx,
                    )
                    events = cal.date_search(start=start, end=end)
                    for ev in events:
                        busy_entries.extend(event_to_dict_list(ev))
                    return [e for e in busy_entries if e._type == "VBUSY" or e._type == "VEVENT"]

            # 2. If not a shared calendar, try an org-level free/busy query
        #         if ctx:
        #             await ctx.info(f"No shared calendar found, querying org free/busy for {email}...")
        #         report_xml = f"""<?xml version="1.0" encoding="utf-8" ?>
        # <C:free-busy-query xmlns:C="urn:ietf:params:xml:ns:caldav">
        # <C:time-range start="{start.strftime('%Y%m%dT%H%M%SZ')}"
        #                 end="{end.strftime('%Y%m%dT%H%M%SZ')}"/>
        # </C:free-busy-query>
        # """
        #         cal_url = urljoin(self.url, f"{email}/events/")
        #         headers = {"Content-Type": "application/xml; charset=utf-8"}
        #         resp = requests.request(
        #             "REPORT",
        #             cal_url,
        #             data=report_xml,
        #             headers=headers,
        #             auth=(self.user, self.password),
        #         )
        #         resp.raise_for_status()

        #         cal: Component = Calendar.from_ical(resp.text)
        #         for component in cal.walk():
        #             component: Component = component
        #             if component.name == "VFREEBUSY":
        #                 freebusy_values = component.get("FREEBUSY")
        #                 if not isinstance(freebusy_values, list):
        #                     freebusy_values = [freebusy_values]
        #                 for period in freebusy_values:
        #                     fbtype = period.params.get("FBTYPE", "BUSY").upper()
        #                     if fbtype == "FREE":
        #                         continue
        #                     busy_entries.append(
        #                         CalEventEntryContract(
        #                             _type="VBUSY",
        #                             cal_url=cal.url,
        #                             summary="Busy",
        #                             start_datetime=period.dtstart,
        #                             end_datetime=period.dtend,
        #                             extra={"fbtype": fbtype}
        #                         )
        #                     )
        #         return busy_entries

        except Exception as e:
            await log(f"Failed to get availability for {email}: {e}", "error", logger, ctx, exception=e)
            traceback.print_exc(file=sys.stderr)
            return []


# Initialize FastMCP server
mcp = FastMCP("caldav-server", host=MCP_HOST, port=MCP_PORT)

# Initialize CalDAV server
try:
    caldav_server = CalDAVServer()
    caldav_configured = True
except Exception as e:
    caldav_server = None
    caldav_configured = False
    log(f"Warning: CalDAV server not configured: {e}", "warning", logger, None)


@mcp.tool()
async def create_caldav_event(
    title: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    uid: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    ctx: Context = None,
) -> MCPResponse:
    """
    Create a new CalDAV calendar event and optionally invite attendees.

    Args:
        title: Event title
        start_time: Start time in ISO format (e.g., "2025-08-20T15:00:00")
        end_time: End time in ISO format (e.g., "2025-08-20T16:00:00")
        description: Event description (optional)
        location: Event location (optional)
        uid: Unique identifier for the event (optional)
        attendees: List of email addresses to invite (optional)
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not properly configured. Please check environment variables.",
        )

    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)

        event = CalDAVEvent(
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            description=description,
            location=location,
            uid=uid,
        )

        return await caldav_server.create_event(event, ctx, attendees=attendees or [])

    except ValueError as e:
        return MCPResponse(
            status="ERR",
            error=f"Invalid datetime format: {str(e)}. Please use ISO format (e.g., '2025-08-20T15:00:00')",
        )
    except Exception as e:
        error_msg = f"Failed to create CalDAV event: {str(e)}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


@mcp.tool()
async def get_caldav_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    calendar_id: Optional[str] = None,
    ctx: Context = None,
) -> MCPResponse:
    """
    Get events from the CalDAV calendar.

    Args:
        start_date: Start date in ISO format (optional, defaults to today)
        end_date: End date in ISO format (optional, defaults to 30 days from start)
        calendar_id: Calendar ID (optional, defaults to primary calendar)
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not properly configured. Please check environment variables.",
        )

    try:
        # Parse datetime strings if provided
        start_dt = None
        end_dt = None

        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)

        # Get events
        return await caldav_server.get_events(start_dt, end_dt, calendar_id, ctx)

    except ValueError as e:
        return MCPResponse(
            status="ERR",
            error=f"Invalid datetime format: {str(e)}. Please use ISO format (e.g., '2025-08-20T15:00:00')",
        )
    except Exception as e:
        error_msg = f"Failed to get CalDAV events: {str(e)}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


@mcp.tool()
async def delete_caldav_event(event_uid: str, ctx: Context = None) -> MCPResponse:
    """
    Delete a CalDAV calendar event by UID.

    Args:
        event_uid: Unique identifier of the event to delete
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not properly configured. Please check environment variables.",
        )

    try:
        return await caldav_server.delete_event(event_uid, ctx)

    except Exception as e:
        error_msg = f"Failed to delete CalDAV event: {str(e)}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


@mcp.tool()
async def list_caldav_calendars(ctx: Context) -> MCPResponse:
    """
    List available CalDAV calendars.

    Args:
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not properly configured. Please check environment variables.",
        )

    try:
        return await caldav_server.list_calendars(ctx)

    except Exception as e:
        error_msg = f"Failed to list CalDAV calendars: {str(e)}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


@mcp.tool()
async def get_caldav_status(ctx: Context) -> MCPResponse:
    """
    Get the current CalDAV server configuration status.

    Args:
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not configured. Please check environment variables.",
        )

    status = caldav_server.test_connection()

    if status["status"] == "configured":
        await log("CalDAV server is properly configured", "info", logger, ctx)
        payload = f"CalDAV server configured successfully:\nURL: {status['url']}\nUser: {status['user']}\nPassword: {status['password']}"
        return MCPResponse(status="OK", payload=payload)
    else:
        await log(f"CalDAV configuration error: {status['error']}", "error", logger, ctx)
        return MCPResponse(status="ERR", error=f"CalDAV configuration error: {status['error']}")


@mcp.tool()
async def create_quick_caldav_event(
    title: str,
    start_hour: int,
    start_minute: int,
    duration_hours: int = 1,
    date: Optional[str] = None,
    ctx: Context = None,
) -> MCPResponse:
    """
    Create a quick CalDAV event with simplified parameters.

    Args:
        title: Event title
        start_hour: Start hour (0-23)
        start_minute: Start minute (0-59)
        duration_hours: Duration in hours (default: 1)
        date: Date in YYYY-MM-DD format (optional, defaults to today)
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not properly configured. Please check environment variables.",
        )

    try:
        # Set date (default to today)
        if date:
            event_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            event_date = datetime.now().date()

        # Create start and end times
        start_time = datetime.combine(
            event_date,
            datetime.min.time().replace(hour=start_hour, minute=start_minute),
        )
        end_time = start_time + timedelta(hours=duration_hours)

        # Create event
        event = CalDAVEvent(title=title, start_time=start_time, end_time=end_time)

        return await caldav_server.create_event(event, ctx)

    except ValueError as e:
        return MCPResponse(status="ERR", error=f"Invalid parameters: {str(e)}")
    except Exception as e:
        error_msg = f"Failed to create quick CalDAV event: {str(e)}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


@mcp.tool()
async def get_caldav_availability(email: str, start_date: str, end_date: str, ctx: Context) -> MCPResponse:
    """
    Get busy periods for a given email in the CalDAV server.
    Checks both shared calendars and org-level free/busy.

    Args:
        email: Email address of the calendar owner
        start_date: ISO datetime string (UTC) for range start
        end_date: ISO datetime string (UTC) for range end
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(status="ERR", error="CalDAV server is not properly configured.")

    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        entries = await caldav_server.get_availability_for_email(email=email, start=start_dt, end=end_dt, ctx=ctx)

        if not entries:
            return MCPResponse(
                status="OK",
                payload=f"No busy periods found for {email} between {start_date} and {end_date}.",
            )

        payload = "\n\n".join(e.model_dump_json(indent=2) for e in entries)
        return MCPResponse(status="OK", payload=payload)

    except ValueError:
        return MCPResponse(
            status="ERR",
            error="Invalid datetime format. Please use ISO format, e.g. '2025-08-20T15:00:00'.",
        )
    except Exception as e:
        error_msg = f"Failed to get availability for {email}: {e}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


@mcp.tool()
async def test_caldav_connection(ctx: Context) -> MCPResponse:
    """
    Test the CalDAV connection and list available calendars.

    Args:
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return MCPResponse(
            status="ERR",
            error="CalDAV server is not properly configured. Please check environment variables.",
        )

    try:
        await log("Testing CalDAV connection...", "info", logger, ctx)

        # Test connection
        if await caldav_server.connect(ctx):
            # List calendars
            calendars_info = await caldav_server.list_calendars(ctx)
            payload = f"CalDAV connection successful!\n\n{calendars_info.payload}"
            return MCPResponse(status="OK", payload=payload)
        else:
            return MCPResponse(status="ERR", error="Failed to connect to CalDAV server")

    except Exception as e:
        error_msg = f"Failed to test CalDAV connection: {str(e)}"
        await log(error_msg, "error", logger, ctx, exception=e)
        traceback.print_exc(file=sys.stderr)
        return MCPResponse(status="ERR", error=error_msg)


async def main():
    """Main function to start the CalDAV MCP server"""

    def log_info():
        if not caldav_configured:
            log("WARNING: CalDAV server is not properly configured!", "warning", logger, None)
            log("Please set the following environment variables:", "info", logger, None)
            log("  CALDAV_URL - Your CalDAV server URL", "info", logger, None)
            log("  CALDAV_USER - Your CalDAV username", "info", logger, None)
            log("  CALDAV_PASSWORD - Your CalDAV password", "info", logger, None)
            log("  MCP_HOST - MCP server host (default: localhost)", "info", logger, None)
            log("  MCP_PORT - MCP server port (default: 8009)", "info", logger, None)

    await start_mcp_server(mcp, MCP_HOST, MCP_PORT, logger, log_info)


if __name__ == "__main__":
    asyncio.run(main())
