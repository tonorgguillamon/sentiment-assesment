from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional
import uuid
from src.models import Sentiment
from pathlib import Path
import json

class Tweet(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())) # generates an unique id for each instace
    tweet: str
    date: datetime
    location: str
    sentiment: Optional[Sentiment] = None

    @validator("date", pre=True)
    # custom validation that processes and transform the date field during the model initialization
    # pre=True ensures that the validation is applied before any other processing (i.e. type coercion)
    def parseDate(cls, value: str) -> datetime:
        return datetime.strptime(value, "%Y-%m-%d")
    
    def __repr__(self):
        return (
            f"Tweet = id:{self.id}, text:{self.tweet}, "
            f"time:{self.date}, location:{self.location} --> Sentiment:{self.sentiment}"
        )

def getDummyTweets(path: Path):
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data