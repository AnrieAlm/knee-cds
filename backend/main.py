from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import google.oauth2.id_token
from google.auth.transport import requests as google_requests
from backend import store
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()

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


def require_user(request: Request):
    id_token = request.cookies.get("token")
    return validate_firebase_token(id_token)


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse(request, "login.html")


@app.get("/")
async def index(request: Request):
    if not require_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(request, "index.html", {"cases": store.list_cases()})



@app.get("/cases/new")
async def new_case_form(request: Request):
    if not require_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(request, "new_case.html")


@app.post("/cases/new")
async def new_case_submit(request: Request):
    if not require_user(request):
        return RedirectResponse("/login")

    form = await request.form()
    store.create_case(form["patient_label"])
    return RedirectResponse("/", status_code=303)

@app.get("/cases/{case_id}")
async def view_case(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return RedirectResponse(f"/cases/{case_id}/history")


@app.get("/cases/{case_id}/history")
async def case_history(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_history.html", {"case": case, "active_tab": "history"})


@app.get("/cases/{case_id}/exam")
async def case_exam(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_exam.html", {"case": case, "active_tab": "exam"})


@app.get("/cases/{case_id}/summary")
async def case_summary(request: Request, case_id: str):
    if not require_user(request):
        return RedirectResponse("/login")

    case = store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return templates.TemplateResponse(request, "case_summary.html", {"case": case, "active_tab": "summary"})

@app.get("/api/status")
async def status():
    return {"status": "ok", "message": "Knee CDS API running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}