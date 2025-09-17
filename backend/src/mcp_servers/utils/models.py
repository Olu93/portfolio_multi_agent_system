


from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MCPResponse(BaseModel):
    status: Literal["OK", "ERR"]
    payload: Optional[str | dict] = None
    error: Optional[str] = None
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())