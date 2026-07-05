from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.oauth2.id_token
from google.auth.transport import requests as google_requests

app = FastAPI(
    title="Knee CDS API",
    description="Agentic clinical decision support for junior physiotherapists",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

firebase_request_adapter = google_requests.Request()


def validate_firebase_token(id_token):
    if not id_token:
        return None
    try:
        return google.oauth2.id_token.verify_firebase_token(id_token, firebase_request_adapter)
    except ValueError as err:
        print(str(err))
        return None


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/")
async def index(request: Request):
    id_token = request.cookies.get("token")
    user_token = validate_firebase_token(id_token)
    if not user_token:
        return RedirectResponse("/login")

    dummy_cases = [
        {"id": 1, "patient_label": "Case #1 — R knee, acute", "created_at": "2026-07-05"},
        {"id": 2, "patient_label": "Case #2 — L knee, chronic", "created_at": "2026-07-04"},
    ]
    return templates.TemplateResponse(request, "index.html", {"cases": dummy_cases})


@app.get("/api/status")
async def status():
    return {"status": "ok", "message": "Knee CDS API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}