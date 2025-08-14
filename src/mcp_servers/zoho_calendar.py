from fastmcp import FastMCP, Context
import os
import asyncio
import traceback
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8008"))

# Zoho Calendar Configuration
ZOHO_ACCESS_TOKEN = os.getenv("ZOHO_ACCESS_TOKEN", "")
ZOHO_CALENDAR_ID = os.getenv("ZOHO_CALENDAR_ID", "primary")
ZOHO_API_BASE = "https://calendar.zoho.com/api/v1"


@dataclass
class CalendarEvent:
    """Represents a calendar event"""
    title: str
    start_time: datetime
    end_time: datetime
    location: Optional[str] = None
    description: Optional[str] = None
    attendees: Optional[List[str]] = None
    timezone: str = "Europe/Amsterdam"
    send_notifications: bool = True


@dataclass
class EventAttendee:
    """Represents an event attendee"""
    email: str
    name: Optional[str] = None
    response_status: Optional[str] = None


class ZohoCalendarServer:
    """Handles Zoho Calendar API operations"""
    
    def __init__(self):
        self.access_token = ZOHO_ACCESS_TOKEN
        self.calendar_id = ZOHO_CALENDAR_ID
        self.api_base = ZOHO_API_BASE
        
        # Validate configuration
        if not self.access_token:
            raise ValueError(
                "ZOHO_ACCESS_TOKEN environment variable must be set"
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for Zoho API requests"""
        return {
            "Authorization": f"Zoho-oauthtoken {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def create_event(self, event: CalendarEvent, ctx: Context) -> str:
        """Create a new calendar event"""
        try:
            await ctx.info(f"Creating event: {event.title}")
            
            # Prepare payload
            payload = {
                "title": event.title,
                "start": {
                    "dateTime": event.start_time.isoformat(),
                    "timeZone": event.timezone
                },
                "end": {
                    "dateTime": event.end_time.isoformat(),
                    "timeZone": event.timezone
                },
                "sendNotifications": event.send_notifications
            }
            
            if event.location:
                payload["location"] = event.location
            
            if event.description:
                payload["description"] = event.description
            
            if event.attendees:
                payload["attendees"] = [{"email": email} for email in event.attendees]
            
            # Make API request
            url = f"{self.api_base}/calendars/{self.calendar_id}/events"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=payload, 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200 or response.status == 201:
                        result = await response.json()
                        await ctx.info("Event created successfully!")
                        return f"Event '{event.title}' created successfully. Event ID: {result.get('id', 'N/A')}"
                    else:
                        error_text = await response.text()
                        await ctx.error(f"Failed to create event. Status: {response.status}")
                        return f"Failed to create event. Status: {response.status}, Response: {error_text}"
                        
        except Exception as e:
            error_msg = f"Failed to create event: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg
    
    async def get_events(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, ctx: Context = None) -> str:
        """Get events from the calendar"""
        try:
            if ctx:
                await ctx.info("Fetching calendar events...")
            
            # Set default date range if not provided
            if not start_date:
                start_date = datetime.now()
            if not end_date:
                end_date = start_date + timedelta(days=30)
            
            # Prepare query parameters
            params = {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            }
            
            url = f"{self.api_base}/calendars/{self.calendar_id}/events"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    params=params, 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        events = await response.json()
                        event_count = len(events.get('events', []))
                        if ctx:
                            await ctx.info(f"Retrieved {event_count} events")
                        return f"Retrieved {event_count} events from {start_date.date()} to {end_date.date()}"
                    else:
                        error_text = await response.text()
                        if ctx:
                            await ctx.error(f"Failed to retrieve events. Status: {response.status}")
                        return f"Failed to retrieve events. Status: {response.status}, Response: {error_text}"
                        
        except Exception as e:
            error_msg = f"Failed to retrieve events: {str(e)}"
            if ctx:
                await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg
    
    async def update_event(self, event_id: str, updates: Dict[str, Any], ctx: Context) -> str:
        """Update an existing calendar event"""
        try:
            await ctx.info(f"Updating event: {event_id}")
            
            url = f"{self.api_base}/calendars/{self.calendar_id}/events/{event_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.put(
                    url, 
                    json=updates, 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200:
                        await ctx.info("Event updated successfully!")
                        return f"Event {event_id} updated successfully"
                    else:
                        error_text = await response.text()
                        await ctx.error(f"Failed to update event. Status: {response.status}")
                        return f"Failed to update event. Status: {response.status}, Response: {error_text}"
                        
        except Exception as e:
            error_msg = f"Failed to update event: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg
    
    async def delete_event(self, event_id: str, ctx: Context) -> str:
        """Delete a calendar event"""
        try:
            await ctx.info(f"Deleting event: {event_id}")
            
            url = f"{self.api_base}/calendars/{self.calendar_id}/events/{event_id}"
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    url, 
                    headers=self._get_headers()
                ) as response:
                    if response.status == 200 or response.status == 204:
                        await ctx.info("Event deleted successfully!")
                        return f"Event {event_id} deleted successfully"
                    else:
                        error_text = await response.text()
                        await ctx.error(f"Failed to delete event. Status: {response.status}")
                        return f"Failed to delete event. Status: {response.status}, Response: {error_text}"
                        
        except Exception as e:
            error_msg = f"Failed to delete event: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Zoho Calendar connection and return status"""
        try:
            return {
                "status": "configured",
                "api_base": self.api_base,
                "calendar_id": self.calendar_id,
                "access_token": f"{self.access_token[:10]}..." if self.access_token else None
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Initialize FastMCP server
mcp = FastMCP("zoho-calendar-server", host=MCP_HOST, port=MCP_PORT)

# Initialize Zoho Calendar server
try:
    calendar_server = ZohoCalendarServer()
    calendar_configured = True
except Exception as e:
    calendar_server = None
    calendar_configured = False
    print(f"Warning: Zoho Calendar server not configured: {e}")


@mcp.tool()
async def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    location: Optional[str] = None,
    description: Optional[str] = None,
    attendees: Optional[List[str]] = None,
    timezone: str = "Europe/Amsterdam",
    send_notifications: bool = True,
    ctx: Context = None
) -> str:
    """
    Create a new calendar event.
    
    Args:
        title: Event title
        start_time: Start time in ISO format (e.g., "2025-08-20T15:00:00")
        end_time: End time in ISO format (e.g., "2025-08-20T16:00:00")
        location: Event location (optional)
        description: Event description (optional)
        attendees: List of attendee email addresses (optional)
        timezone: Timezone for the event (default: Europe/Amsterdam)
        send_notifications: Whether to send notifications (default: True)
        ctx: MCP context for logging
    """
    if not calendar_configured:
        return "Zoho Calendar server is not properly configured. Please check environment variables."
    
    try:
        # Parse datetime strings
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        
        # Create event object
        event = CalendarEvent(
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            location=location,
            description=description,
            attendees=attendees,
            timezone=timezone,
            send_notifications=send_notifications
        )
        
        # Create event
        return await calendar_server.create_event(event, ctx)
        
    except ValueError as e:
        return f"Invalid datetime format: {str(e)}. Please use ISO format (e.g., '2025-08-20T15:00:00')"
    except Exception as e:
        error_msg = f"Failed to create calendar event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def get_calendar_events(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Get events from the calendar.
    
    Args:
        start_date: Start date in ISO format (optional, defaults to today)
        end_date: End date in ISO format (optional, defaults to 30 days from start)
        ctx: MCP context for logging
    """
    if not calendar_configured:
        return "Zoho Calendar server is not properly configured. Please check environment variables."
    
    try:
        # Parse datetime strings if provided
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        
        # Get events
        return await calendar_server.get_events(start_dt, end_dt, ctx)
        
    except ValueError as e:
        return f"Invalid datetime format: {str(e)}. Please use ISO format (e.g., '2025-08-20T15:00:00')"
    except Exception as e:
        error_msg = f"Failed to get calendar events: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def update_calendar_event(
    event_id: str,
    updates: Dict[str, Any],
    ctx: Context = None
) -> str:
    """
    Update an existing calendar event.
    
    Args:
        event_id: ID of the event to update
        updates: Dictionary of fields to update
        ctx: MCP context for logging
    """
    if not calendar_configured:
        return "Zoho Calendar server is not properly configured. Please check environment variables."
    
    try:
        return await calendar_server.update_event(event_id, updates, ctx)
        
    except Exception as e:
        error_msg = f"Failed to update calendar event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def delete_calendar_event(
    event_id: str,
    ctx: Context = None
) -> str:
    """
    Delete a calendar event.
    
    Args:
        event_id: ID of the event to delete
        ctx: MCP context for logging
    """
    if not calendar_configured:
        return "Zoho Calendar server is not properly configured. Please check environment variables."
    
    try:
        return await calendar_server.delete_event(event_id, ctx)
        
    except Exception as e:
        error_msg = f"Failed to delete calendar event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


@mcp.tool()
async def get_calendar_status(ctx: Context) -> str:
    """
    Get the current Zoho Calendar server configuration status.
    
    Args:
        ctx: MCP context for logging
    """
    if not calendar_configured:
        return "Zoho Calendar server is not configured. Please check environment variables."
    
    status = calendar_server.test_connection()
    
    if status["status"] == "configured":
        await ctx.info("Zoho Calendar server is properly configured")
        return f"Zoho Calendar server configured successfully:\n" \
               f"API Base: {status['api_base']}\n" \
               f"Calendar ID: {status['calendar_id']}\n" \
               f"Access Token: {status['access_token']}"
    else:
        await ctx.error(f"Zoho Calendar configuration error: {status['error']}")
        return f"Zoho Calendar configuration error: {status['error']}"


@mcp.tool()
async def create_quick_event(
    title: str,
    start_hour: int,
    start_minute: int,
    duration_hours: int = 1,
    date: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    Create a quick event with simplified parameters.
    
    Args:
        title: Event title
        start_hour: Start hour (0-23)
        start_minute: Start minute (0-59)
        duration_hours: Duration in hours (default: 1)
        date: Date in YYYY-MM-DD format (optional, defaults to today)
        ctx: MCP context for logging
    """
    if not calendar_configured:
        return "Zoho Calendar server is not properly configured. Please check environment variables."
    
    try:
        # Set date (default to today)
        if date:
            event_date = datetime.strptime(date, "%Y-%m-%d").date()
        else:
            event_date = datetime.now().date()
        
        # Create start and end times
        start_time = datetime.combine(event_date, datetime.min.time().replace(hour=start_hour, minute=start_minute))
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Create event
        event = CalendarEvent(
            title=title,
            start_time=start_time,
            end_time=end_time
        )
        
        return await calendar_server.create_event(event, ctx)
        
    except ValueError as e:
        return f"Invalid parameters: {str(e)}"
    except Exception as e:
        error_msg = f"Failed to create quick event: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg


if __name__ == "__main__":
    print("=== Starting Zoho Calendar MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    
    if not calendar_configured:
        print("WARNING: Zoho Calendar server is not properly configured!")
        print("Please set the following environment variables:")
        print("  ZOHO_ACCESS_TOKEN - Your Zoho OAuth access token")
        print("  ZOHO_CALENDAR_ID - Calendar ID (default: primary)")
        print("  MCP_HOST - MCP server host (default: localhost)")
        print("  MCP_PORT - MCP server port (default: 8008)")
    
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        print("=== Zoho Calendar MCP Server shutting down ===")
