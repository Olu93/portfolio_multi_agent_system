from fastmcp import FastMCP, Context
import os
import asyncio
import traceback
import sys
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
from caldav import DAVClient
from caldav.elements import dav, cdav
from dotenv import load_dotenv, find_dotenv
from caldav.calendarobjectresource import Event
from caldav.collection import Calendar as CalDAVCalendar
from icalendar import Calendar
from icalendar.cal import Component
from icalendar.prop import vDDDTypes, TimeBase, vCategory
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv(find_dotenv())

# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8009"))

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


class CalEventEntryContract(BaseModel):
    _type: Literal["VEVENT", "VBUSY"]
    summary: str = Field("", description="The summary of the event")
    start_datetime: datetime = Field(
        datetime.now(), description="The start datetime of the event as datetime object"
    )
    end_datetime: datetime = Field(
        datetime.now(), description="The end datetime of the event as datetime object"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Extra fields of the event"
    )


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

            for period in freebusy_values:
                fbtype = period.params.get("FBTYPE", "BUSY").upper()
                if fbtype == "FREE":
                    continue  # skip free times

                busy_event = CalEventEntryContract(
                    _type="VBUSY",
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
            raise ValueError(
                "CALDAV_URL, CALDAV_USER, and CALDAV_PASSWORD environment variables must be set"
            )

    async def connect(self, ctx: Context) -> bool:
        """Connect to CalDAV server and initialize calendars"""
        try:
            await ctx.info("Connecting to CalDAV server...")

            # Create client and connect
            self.client = DAVClient(
                url=self.url, username=self.user, password=self.password
            )
            self.principal = self.client.principal()
            self.calendars = self.principal.calendars()

            if not self.calendars:
                await ctx.error("No calendars found on CalDAV server")
                return False

            # Set primary calendar (first one)
            self.primary_calendar = self.calendars[0]
            await ctx.info(
                f"Connected to CalDAV server. Found {len(self.calendars)} calendars"
            )
            await ctx.info(f"Primary calendar: {self.primary_calendar.name}")

            return True

        except Exception as e:
            await ctx.error(f"Failed to connect to CalDAV server: {str(e)}")
            return False

    def _generate_ical_event(self, event: CalDAVEvent) -> str:
        """Generate iCal format for an event"""
        # Generate unique UID if not provided
        uid = event.uid or f"{int(datetime.utcnow().timestamp())}@caldav.mcp"

        # Format dates in iCal format
        dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dtstart = event.start_time.strftime("%Y%m%dT%H%M%SZ")
        dtend = event.end_time.strftime("%Y%m%dT%H%M%SZ")

        ical = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//CalDAV MCP Server//EN
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:{event.title}"""

        if event.description:
            ical += f"\nDESCRIPTION:{event.description}"

        if event.location:
            ical += f"\nLOCATION:{event.location}"

        ical += "\nEND:VEVENT\nEND:VCALENDAR"

        return ical

    async def create_event(self, event: CalDAVEvent, ctx: Context) -> str:
        """Create a new calendar event"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return "Failed to connect to CalDAV server"

            await ctx.info(f"Creating event: {event.title}")

            # Generate iCal format
            ical_content = self._generate_ical_event(event)

            # Add event to calendar
            self.primary_calendar.add_event(ical_content)

            await ctx.info("Event created successfully!")
            return f"Event '{event.title}' created successfully in CalDAV calendar"

        except Exception as e:
            error_msg = f"Failed to create event: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg

    # TODO: Add limit parameter to get events but keep the default to 10
    async def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ctx: Context = None,
    ) -> str:
        """Get events from the calendar"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return "Failed to connect to CalDAV server"

            if ctx:
                await ctx.info("Fetching calendar events...")

            # Set default date range if not provided
            if not start_date:
                start_date = datetime.utcnow()
            if not end_date:
                end_date = start_date + timedelta(days=30)

            # Search for events in date range
            results = self.primary_calendar.date_search(start=start_date, end=end_date)

            if ctx:
                await ctx.info(f"Retrieved {len(results)} events")

            # Format event details
            event_details: list[CalEventEntryContract] = []
            for ev in results[:10]:  # Limit to first 10 events
                try:
                    event_details.extend(event_to_dict_list(ev))
                except Exception as e:
                    error_event = CalEventEntryContract(
                        _type="VEVENT",
                        summary="Error parsing event",
                        start_datetime=ev.dtstart,
                        end_datetime=ev.dtend,
                    )
                    error_event.extra["error"] = str(e)
                    event_details.append(error_event)

            return (
                f"Retrieved {len(results)} events from {start_date.date()} to {end_date.date()}:\n\n"
                + "\n\n".join([event.model_dump_json(indent=2) for event in event_details])
            )

        except Exception as e:
            error_msg = f"Failed to retrieve events: {str(e)}"
            if ctx:
                await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg

    async def delete_event(self, event_uid: str, ctx: Context) -> str:
        """Delete a calendar event by UID"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return "Failed to connect to CalDAV server"

            await ctx.info(f"Searching for event with UID: {event_uid}")

            # Search for events in a wide date range to find the specific event
            start_date = datetime.utcnow() - timedelta(days=365)
            end_date = datetime.utcnow() + timedelta(days=365)

            results = self.primary_calendar.date_search(start=start_date, end=end_date)

            # Find event with matching UID
            target_event = None
            for ev in results:
                try:
                    if (
                        hasattr(ev.instance.vevent, "uid")
                        and ev.instance.vevent.uid.value == event_uid
                    ):
                        target_event = ev
                        break
                except Exception:
                    continue

            if not target_event:
                return f"Event with UID {event_uid} not found"

            # Delete the event
            target_event.delete()
            await ctx.info("Event deleted successfully!")
            return f"Event with UID {event_uid} deleted successfully"

        except Exception as e:
            error_msg = f"Failed to delete event: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg

    async def list_calendars(self, ctx: Context) -> str:
        """List available calendars"""
        try:
            if not self.primary_calendar:
                if not await self.connect(ctx):
                    return "Failed to connect to CalDAV server"

            calendar_list: list[str] = []
            for i, cal in enumerate(self.calendars):
                calendar_list.append(
                    f"{i+1}. Name: {cal.name} - ID: {cal.id} - URL: {cal.url}"
                )

            return f"Available calendars:\n" + "\n".join(calendar_list)

        except Exception as e:
            error_msg = f"Failed to list calendars: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg

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


# Initialize FastMCP server
mcp = FastMCP("caldav-server", host=MCP_HOST, port=MCP_PORT)

# Initialize CalDAV server
try:
    caldav_server = CalDAVServer()
    caldav_configured = True
except Exception as e:
    caldav_server = None
    caldav_configured = False
    print(f"Warning: CalDAV server not configured: {e}")


@mcp.tool()
async def create_caldav_event(
    title: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    uid: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Create a new CalDAV calendar event.

    Args:
        title: Event title
        start_time: Start time in ISO format (e.g., "2025-08-20T15:00:00")
        end_time: End time in ISO format (e.g., "2025-08-20T16:00:00")
        description: Event description (optional)
        location: Event location (optional)
        uid: Unique identifier for the event (optional, auto-generated if not provided)
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return "CalDAV server is not properly configured. Please check environment variables."

    try:
        # Parse datetime strings
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)

        # Create event object
        event = CalDAVEvent(
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            description=description,
            location=location,
            uid=uid,
        )

        # Create event
        return await caldav_server.create_event(event, ctx)

    except ValueError as e:
        return f"Invalid datetime format: {str(e)}. Please use ISO format (e.g., '2025-08-20T15:00:00')"
    except Exception as e:
        error_msg = f"Failed to create CalDAV event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def get_caldav_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Get events from the CalDAV calendar.

    Args:
        start_date: Start date in ISO format (optional, defaults to today)
        end_date: End date in ISO format (optional, defaults to 30 days from start)
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return "CalDAV server is not properly configured. Please check environment variables."

    try:
        # Parse datetime strings if provided
        start_dt = None
        end_dt = None

        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)

        # Get events
        return await caldav_server.get_events(start_dt, end_dt, ctx)

    except ValueError as e:
        return f"Invalid datetime format: {str(e)}. Please use ISO format (e.g., '2025-08-20T15:00:00')"
    except Exception as e:
        error_msg = f"Failed to get CalDAV events: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def delete_caldav_event(event_uid: str, ctx: Context = None) -> str:
    """
    Delete a CalDAV calendar event by UID.

    Args:
        event_uid: Unique identifier of the event to delete
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return "CalDAV server is not properly configured. Please check environment variables."

    try:
        return await caldav_server.delete_event(event_uid, ctx)

    except Exception as e:
        error_msg = f"Failed to delete CalDAV event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def list_caldav_calendars(ctx: Context) -> str:
    """
    List available CalDAV calendars.

    Args:
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return "CalDAV server is not properly configured. Please check environment variables."

    try:
        return await caldav_server.list_calendars(ctx)

    except Exception as e:
        error_msg = f"Failed to list CalDAV calendars: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def get_caldav_status(ctx: Context) -> str:
    """
    Get the current CalDAV server configuration status.

    Args:
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return "CalDAV server is not configured. Please check environment variables."

    status = caldav_server.test_connection()

    if status["status"] == "configured":
        await ctx.info("CalDAV server is properly configured")
        return (
            f"CalDAV server configured successfully:\n"
            f"URL: {status['url']}\n"
            f"User: {status['user']}\n"
            f"Password: {status['password']}"
        )
    else:
        await ctx.error(f"CalDAV configuration error: {status['error']}")
        return f"CalDAV configuration error: {status['error']}"


@mcp.tool()
async def create_quick_caldav_event(
    title: str,
    start_hour: int,
    start_minute: int,
    duration_hours: int = 1,
    date: Optional[str] = None,
    ctx: Context = None,
) -> str:
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
        return "CalDAV server is not properly configured. Please check environment variables."

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
        return f"Invalid parameters: {str(e)}"
    except Exception as e:
        error_msg = f"Failed to create quick CalDAV event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def test_caldav_connection(ctx: Context) -> str:
    """
    Test the CalDAV connection and list available calendars.

    Args:
        ctx: MCP context for logging
    """
    if not caldav_configured:
        return "CalDAV server is not properly configured. Please check environment variables."

    try:
        await ctx.info("Testing CalDAV connection...")

        # Test connection
        if await caldav_server.connect(ctx):
            # List calendars
            calendars_info = await caldav_server.list_calendars(ctx)
            return f"CalDAV connection successful!\n\n{calendars_info}"
        else:
            return "Failed to connect to CalDAV server"

    except Exception as e:
        error_msg = f"Failed to test CalDAV connection: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


if __name__ == "__main__":
    print("=== Starting CalDAV MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")

    if not caldav_configured:
        print("WARNING: CalDAV server is not properly configured!")
        print("Please set the following environment variables:")
        print("  CALDAV_URL - Your CalDAV server URL")
        print("  CALDAV_USER - Your CalDAV username")
        print("  CALDAV_PASSWORD - Your CalDAV password")
        print("  MCP_HOST - MCP server host (default: localhost)")
        print("  MCP_PORT - MCP server port (default: 8009)")

    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        print("=== CalDAV MCP Server shutting down ===")
