import asyncio
from typing import Dict

from api import models
from config import settings
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic.main import BaseModel
from sqlalchemy.orm import Session

from .database import SessionLocal, engine
from .models import Phrases

# Create db & table with SQLalchemy
models.Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic classes
class PhraseRequest(BaseModel):
    phrase: str
    #    probabilities: Dict[str, float]
    sentiment: str
    confidence: float


# FastAPI
app = FastAPI(
    title=settings.name, description=settings.description, version=settings.version
)

# Jinja2Templates
templates = Jinja2Templates(
    directory="D:/CompSci/Projects/sentiment-analysis/sentiment_analysis/templates/"
)

# NLP classification background_task
def predict_phrase_sentiment(id: int):
    """
    ,
    """
    db = SessionLocal()
    phrase = db.query(Phrases).filter(Phrases.id == id).first()

    phrase.sentiment = "positive"
    db.add(phrase)
    db.commit()


# Routes
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request},
    )


@app.post("/phrase")
def create_phrase(
    phrase_request: PhraseRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    ,
    (2) add database record
    (3) give background_tasks a reference of phrase record
    """
    phrase = Phrases()
    phrase.phrase = phrase_request.phrase

    db.add(phrase)
    db.commit()

    background_tasks.add_task(predict_phrase_sentiment, phrase.id)

    return {
        "code": "success",
        "message": "phrase added",
    }
