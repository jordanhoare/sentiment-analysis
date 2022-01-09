from typing import Dict

from api import models
from config import settings
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from nlp_classifier.two_classifier import BertClassifier
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
class SentimentRequest(BaseModel):
    phrase: str


class SentimentResponse(BaseModel):
    probabilities: Dict[str, float]
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
def predict_phrase_sentiment(
    id: int,
):
    """
    ,NLP_classifier: Classifier = Depends(get_model), ^^
    """
    db = SessionLocal()
    phrase = db.query(Phrases).filter(Phrases.id == id).first()
    positive_score, neutral_score, negative_score = BertClassifier(phrase).return_list()
    phrase.positive = positive_score
    # phrase.negative = negative_score
    # phrase.neutral = neutral_score
    # phrase.probabilities = probabilities
    # phrase.confidence = confidence
    # phrase.sentiment = sentiment
    phrase.sentiment = "sentiment"
    db.add(phrase)
    db.commit()


# Routes
@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    ,
    """
    phrases = db.query(Phrases).all()
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "phrases": phrases,
        },
    )


@app.post("/phrase")
def create_phrase(
    phrase_request: SentimentRequest,
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
