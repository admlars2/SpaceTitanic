from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    NASA_API_KEY: str = Field(..., env="NASA_API_KEY")