from config import settings
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Jinja2Templates
templates = Jinja2Templates(
    directory="D:/CompSci/Projects/sentiment-analysis/sentiment_analysis/templates/"
)

# FastAPI
app = FastAPI(
    title=settings.name, description=settings.description, version=settings.version
)

# Routes
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/phrase")
def create_phrase():
    return {"code": "success", "message": "stock created"}
