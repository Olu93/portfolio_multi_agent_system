from typing import Optional
from pydantic import BaseModel, Field

class RealEstate(BaseModel):
    file_name: str = Field(..., description="The name of the file to store the data in. Consists of the address without any special characters and whitespaces replaced with underscores. The file ending is .json")
    url: str = Field(..., description="The URL of the property.")  
    title: str = Field(..., description="The title of the property.")
    description: Optional[str] = Field(None, description="The description of the property.")
    price: Optional[str] = Field(None, description="The price of the property.")
    address: str = Field(..., description="The address of the property.")
    bedrooms: Optional[int] = Field(None, description="The number of bedrooms in the property.")
    bathrooms: Optional[int] = Field(None, description="The number of bathrooms in the property.")
    sqft: Optional[int] = Field(None, description="The square footage of the property.")
    image_url: Optional[str] = Field(None, description="The URL of the image of the property.")
    extra_data: Optional[dict] = Field(None, description="Extra data about the property.")