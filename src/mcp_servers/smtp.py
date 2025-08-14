from fastmcp import FastMCP, Context
import os
import asyncio
import traceback
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv, find_dotenv
import aiosmtplib

# Load environment variables
load_dotenv(find_dotenv())

# Get configuration from environment variables
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8002"))

# SMTP Configuration
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
# SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
# SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", SMTP_USERNAME)


@dataclass
class EmailAttachment:
    """Represents an email attachment"""
    file_path: str
    filename: Optional[str] = None
    content_type: Optional[str] = None


@dataclass
class EmailMessage:
    """Represents an email message"""
    to_emails: List[str]
    subject: str
    body: str
    body_type: str = "plain"  # "plain" or "html"
    from_email: Optional[str] = None
    cc_emails: Optional[List[str]] = None
    bcc_emails: Optional[List[str]] = None
    attachments: Optional[List[EmailAttachment]] = None


class SMTPServer:
    """Handles SMTP operations for sending emails"""
    
    def __init__(self):
        self.host = SMTP_HOST
        self.port = SMTP_PORT
        self.username = SMTP_USERNAME
        self.password = SMTP_PASSWORD
        # self.use_tls = SMTP_USE_TLS
        # self.use_ssl = SMTP_USE_SSL
        
        # Validate configuration
        if not self.username or not self.password:
            raise ValueError(
                "SMTP_USERNAME and SMTP_PASSWORD environment variables must be set"
            )
    
    async def send_email(self, email_msg: EmailMessage, ctx: Context) -> str:
        """Send an email with optional attachments"""
        try:
            await ctx.info(f"Preparing to send email to: {', '.join(email_msg.to_emails)}")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_msg.from_email or DEFAULT_FROM_EMAIL
            msg['To'] = ', '.join(email_msg.to_emails)
            msg['Subject'] = email_msg.subject
            
            if email_msg.cc_emails:
                msg['Cc'] = ', '.join(email_msg.cc_emails)
            
            # Add body
            if email_msg.body_type.lower() == "html":
                msg.attach(MIMEText(email_msg.body, 'html'))
            else:
                msg.attach(MIMEText(email_msg.body, 'plain'))
            
            # Add attachments
            if email_msg.attachments:
                await ctx.info(f"Processing {len(email_msg.attachments)} attachments")
                for attachment in email_msg.attachments:
                    await self._add_attachment(msg, attachment, ctx)
            
            # Prepare recipient list
            recipients = email_msg.to_emails.copy()
            if email_msg.cc_emails:
                recipients.extend(email_msg.cc_emails)
            if email_msg.bcc_emails:
                recipients.extend(email_msg.bcc_emails)
            
            # Send email
            await ctx.info("Connecting to SMTP server...")
            
            await aiosmtplib.send(
                    msg,
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    use_tls=True,
                    start_tls=False,
                    recipients=recipients
                )
            
            await ctx.info("Email sent successfully!")
            return f"Email sent successfully to {', '.join(email_msg.to_emails)}"
            
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            await ctx.error(error_msg)
            traceback.print_exc(file=sys.stderr)
            return error_msg
    
    async def _add_attachment(self, msg: MIMEMultipart, attachment: EmailAttachment, ctx: Context):
        """Add an attachment to the email message"""
        try:
            file_path = Path(attachment.file_path)
            
            if not file_path.exists():
                await ctx.warning(f"Attachment file not found: {attachment.file_path}")
                return
            
            # Determine content type
            content_type = attachment.content_type
            if not content_type:
                content_type, _ = mimetypes.guess_type(str(file_path))
                if not content_type:
                    content_type = 'application/octet-stream'
            
            # Determine filename
            filename = attachment.filename or file_path.name
            
            # Read file and create attachment
            with open(file_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            
            msg.attach(part)
            await ctx.info(f"Added attachment: {filename}")
            
        except Exception as e:
            await ctx.error(f"Failed to add attachment {attachment.file_path}: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test SMTP connection and return status"""
        try:
            # This is a basic validation - actual connection test would require async
            return {
                "status": "configured",
                "host": self.host,
                "port": self.port,
                "username": self.username,
                # "use_tls": self.use_tls,
                # "use_ssl": self.use_ssl,
                "from_email": DEFAULT_FROM_EMAIL
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


# Initialize FastMCP server
mcp = FastMCP("smtp-server", host=MCP_HOST, port=MCP_PORT)

# Initialize SMTP server
try:
    smtp_server = SMTPServer()
    smtp_configured = True
except Exception as e:
    smtp_server = None
    smtp_configured = False
    print(f"Warning: SMTP server not configured: {e}")



async def _send_email(
    to_emails: List[str],
    subject: str,
    body: str,
    body_type: str = "plain",
    from_email: Optional[str] = None,
    cc_emails: Optional[List[str]] = None,
    bcc_emails: Optional[List[str]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    ctx: Context = None
) -> str:
    """
    Send an email with optional attachments.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        body: Email body content
        body_type: Type of body content ("plain" or "html")
        from_email: Sender email address (optional, uses default if not specified)
        cc_emails: List of CC recipient email addresses (optional)
        bcc_emails: List of BCC recipient email addresses (optional)
        attachments: List of attachment dictionaries with keys: file_path, filename (optional), content_type (optional)
        ctx: MCP context for logging
    """
    if not smtp_configured:
        return "SMTP server is not properly configured. Please check environment variables."
    
    try:
        # Convert attachment dictionaries to EmailAttachment objects
        email_attachments = None
        if attachments:
            email_attachments = []
            for att_dict in attachments:
                email_attachments.append(EmailAttachment(
                    file_path=att_dict.get("file_path"),
                    filename=att_dict.get("filename"),
                    content_type=att_dict.get("content_type")
                ))
        
        # Create email message
        email_msg = EmailMessage(
            to_emails=to_emails,
            subject=subject,
            body=body,
            body_type=body_type,
            from_email=from_email,
            cc_emails=cc_emails,
            bcc_emails=bcc_emails,
            attachments=email_attachments
        )
        
        # Send email
        return await smtp_server.send_email(email_msg, ctx)
        
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        traceback.print_exc(file=sys.stderr)
        return error_msg



@mcp.tool()
async def send_email(
    to_emails: List[str],
    subject: str,
    body: str,
    body_type: str = "plain",
    from_email: Optional[str] = None,
    cc_emails: Optional[List[str]] = None,
    bcc_emails: Optional[List[str]] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    ctx: Context = None
) -> str:
    """
    Send an email with optional attachments.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        body: Email body content
        body_type: Type of body content ("plain" or "html")
        from_email: Sender email address (optional, uses default if not specified)
        cc_emails: List of CC recipient email addresses (optional)
        bcc_emails: List of BCC recipient email addresses (optional)
        attachments: List of attachment dictionaries with keys: file_path, filename (optional), content_type (optional)
        ctx: MCP context for logging
    """
    return await _send_email(
        to_emails=to_emails,
        subject=subject,
        body=body,
        body_type=body_type,
        from_email=from_email,
        cc_emails=cc_emails,
        bcc_emails=bcc_emails,
        attachments=attachments,
        ctx=ctx
    )


@mcp.tool()
async def send_simple_email(
    to_email: str,
    subject: str,
    body: str,
    ctx: Context = None
) -> str:
    """
    Send a simple email to a single recipient.
    
    Args:
        to_email: Recipient email address
        subject: Email subject line
        body: Email body content
        ctx: MCP context for logging
    """
    return await _send_email(
        to_emails=[to_email],
        subject=subject,
        body=body,
        ctx=ctx
    )


@mcp.tool()
async def send_html_email(
    to_emails: List[str],
    subject: str,
    html_body: str,
    from_email: Optional[str] = None,
    attachments: Optional[List[Dict[str, str]]] = None,
    ctx: Context = None
) -> str:
    """
    Send an HTML email with optional attachments.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject line
        html_body: HTML formatted email body
        from_email: Sender email address (optional)
        attachments: List of attachment dictionaries
        ctx: MCP context for logging
    """
    return await _send_email(
        to_emails=to_emails,
        subject=subject,
        body=html_body,
        body_type="html",
        from_email=from_email,
        attachments=attachments,
        ctx=ctx
    )


@mcp.tool()
async def get_smtp_status(ctx: Context) -> str:
    """
    Get the current SMTP server configuration status.
    
    Args:
        ctx: MCP context for logging
    """
    if not smtp_configured:
        return "SMTP server is not configured. Please check environment variables."
    
    status = smtp_server.test_connection()
    
    if status["status"] == "configured":
        await ctx.info("SMTP server is properly configured")
        return f"SMTP server configured successfully:\n" \
               f"Host: {status['host']}\n" \
               f"Port: {status['port']}\n" \
               f"Username: {status['username']}\n" \
               f"Use TLS: {status['use_tls']}\n" \
               f"Use SSL: {status['use_ssl']}\n" \
               f"Default From: {status['from_email']}"
    else:
        await ctx.error(f"SMTP configuration error: {status['error']}")
        return f"SMTP configuration error: {status['error']}"


@mcp.tool()
async def validate_email_address(email: str, ctx: Context) -> str:
    """
    Basic email address validation.
    
    Args:
        email: Email address to validate
        ctx: MCP context for logging
    """
    import re
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(pattern, email):
        await ctx.info(f"Email address '{email}' is valid")
        return f"Email address '{email}' is valid"
    else:
        await ctx.warning(f"Email address '{email}' is invalid")
        return f"Email address '{email}' is invalid"


if __name__ == "__main__":
    print("=== Starting SMTP MCP Server ===")
    print(f"Server will run on {MCP_HOST}:{MCP_PORT}")
    
    if not smtp_configured:
        print("WARNING: SMTP server is not properly configured!")
        print("Please set the following environment variables:")
        print("  SMTP_HOST - SMTP server hostname")
        print("  SMTP_PORT - SMTP server port")
        print("  SMTP_USERNAME - SMTP username/email")
        print("  SMTP_PASSWORD - SMTP password/app password")
        print("  SMTP_USE_TLS - Use STARTTLS (true/false)")
        print("  SMTP_USE_SSL - Use SSL (true/false)")
        print("  DEFAULT_FROM_EMAIL - Default sender email")
        print("  MCP_HOST - MCP server host (default: localhost)")
        print("  MCP_PORT - MCP server port (default: 8002)")
    
    try:
        print("Server initialized and ready to handle connections")
        mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"Server crashed: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        print("=== SMTP MCP Server shutting down ===")
