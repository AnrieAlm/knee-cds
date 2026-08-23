"""
backend/investigation_extract.py

Structured extraction of radiology REPORT text into fixed fields.

SCOPE BOUNDARY - read this before modifying:
This module performs document understanding only. It transcribes text that
a radiologist has already written. It does not interpret images, and it is
never given an anatomical image study to reason about. If an upload turns
out to be a radiograph or MRI slice rather than a typed report, extraction
is REJECTED in Python and the clinician is asked to enter the reported
findings manually.

Routing:
  PDF with a text layer  -> text extracted locally, structured by the text
                            model. No vision model involved.
  PDF without text layer -> first pages rasterised, vision model used as OCR.
  PNG / JPEG             -> vision model used as OCR.

Output is always written with extraction_status = EXTRACTED (or FAILED /
REJECTED). It is never written as VERIFIED. Only a clinician action can
set VERIFIED, and only VERIFIED / MANUAL records reach the agent
(see investigation_context.build_investigation_context).
"""

import base64
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from backend.constants import (
    GEN_MODEL,
    INV_ALLOWED_UPLOAD_TYPES,
    INV_DEFAULT_BODY_PART,
    INV_EXTRACTION_MODEL,
    INV_MAX_FINDINGS_CHARS,
    INV_MAX_IMPRESSION_CHARS,
    INV_MAX_UPLOAD_BYTES,
    INV_SOURCE_MANUAL_ENTRY,
    INV_SOURCE_UPLOADED_REPORT,
    INV_STATUS_EXTRACTED,
    INV_STATUS_FAILED,
    INV_STATUS_MANUAL,
    INV_STATUS_REJECTED,
    MODALITIES,
    MODALITY_CT,
    MODALITY_MRI,
    MODALITY_OTHER,
    MODALITY_ULTRASOUND,
    MODALITY_XRAY,
    SIDE_BILATERAL,
    SIDE_LEFT,
    SIDE_NOT_STATED,
    SIDE_RIGHT,
    SIDES,
)

logger = logging.getLogger(__name__)

# Text model already used elsewhere in the system.
#TEXT_MODEL = "llama-3.3-70b-versatile"
TEXT_MODEL = GEN_MODEL
# A PDF page carrying fewer characters than this is treated as a scan.
_TEXT_LAYER_MIN_CHARS = 120

# Rasterisation caps - a report is one or two pages; more than this is not
# a report we should be trying to parse.
_MAX_RASTER_PAGES = 2
_RASTER_ZOOM = 2.0  # ~144 dpi, enough for OCR of printed text


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a transcription tool for a clinical documentation \
system. You transcribe text from radiology REPORTS into structured fields.

Absolute rules:
- Transcribe only. Never interpret, diagnose, summarise, or add clinical \
opinion of your own.
- Never describe the appearance of any anatomy. You are reading a typed \
report, not an image study.
- If a field is not stated in the document, return null. Never guess, and \
never infer a value from context.
- If the document is not a radiology report - for example it is a radiograph, \
an MRI slice, a referral letter, a prescription, or an unrelated document - \
set is_report to false and set every other field to null.

Respond with a single JSON object and nothing else. No preamble, no \
explanation, no markdown fences.

Schema:
{
  "is_report": boolean,
  "document_kind": string or null,
  "modality": one of "XRAY", "MRI", "CT", "ULTRASOUND", "OTHER", or null,
  "body_part": string or null,
  "side": one of "LEFT", "RIGHT", "BILATERAL", or null,
  "study_date": string in YYYY-MM-DD format, or null,
  "report_findings": string or null,
  "report_impression": string or null
}

report_findings must be the findings or body section of the report, \
transcribed verbatim. report_impression must be the impression, conclusion \
or opinion section, transcribed verbatim. If the report is not divided into \
sections, put the whole body in report_findings and leave report_impression \
null."""

_USER_INSTRUCTION = (
    "Transcribe this document into the JSON schema. If it is not a radiology "
    "report, set is_report to false."
)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

class UploadRejected(Exception):
    """Raised for uploads that should never reach the model at all."""


def validate_upload(file_bytes: bytes, content_type: Optional[str]) -> None:
    if not file_bytes:
        raise UploadRejected("The uploaded file is empty.")

    if len(file_bytes) > INV_MAX_UPLOAD_BYTES:
        limit_mb = INV_MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadRejected(
            f"File is too large. The limit is {limit_mb} MB."
        )

    if content_type not in INV_ALLOWED_UPLOAD_TYPES:
        raise UploadRejected(
            "Unsupported file type. Upload a PDF, PNG or JPEG of the "
            "radiology report."
        )


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def _pdf_text_layer(file_bytes: bytes) -> Optional[str]:
    """Return the PDF's text layer, or None if it looks like a scan."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pages = [doc[i].get_text("text") for i in range(min(len(doc), 4))]
    except Exception:
        logger.exception("Could not open uploaded PDF")
        return None

    text = "\n".join(p for p in pages if p).strip()
    return text if len(text) >= _TEXT_LAYER_MIN_CHARS else None


def _pdf_to_pngs(file_bytes: bytes) -> List[bytes]:
    """Rasterise the first pages of a scanned PDF for OCR."""
    images: List[bytes] = []
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            matrix = fitz.Matrix(_RASTER_ZOOM, _RASTER_ZOOM)
            for index in range(min(len(doc), _MAX_RASTER_PAGES)):
                pixmap = doc[index].get_pixmap(matrix=matrix)
                images.append(pixmap.tobytes("png"))
    except Exception:
        logger.exception("Could not rasterise uploaded PDF")
        return []
    return images


def _data_url(image_bytes: bytes, media_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _call_text_model(groq_client, document_text: str) -> str:
    response = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{_USER_INSTRUCTION}\n\n---\n{document_text}\n---",
            },
        ],
    )
    return response.choices[0].message.content


def _call_vision_model(groq_client, images: List[Tuple[bytes, str]]) -> str:
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": _USER_INSTRUCTION}
    ]
    for image_bytes, media_type in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _data_url(image_bytes, media_type)},
        })

    response = groq_client.chat.completions.create(
        model=INV_EXTRACTION_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Response parsing and normalisation
# ---------------------------------------------------------------------------

def _parse_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Extraction returned non-JSON output")
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalise_modality(value: Any) -> str:
    if not isinstance(value, str):
        return MODALITY_OTHER
    token = value.strip().upper()
    if token in MODALITIES:
        return token
    # Tolerate common report wordings without letting the model invent values.
    if any(k in token for k in ("MAGNETIC", "MR ")):
        return MODALITY_MRI
    if any(k in token for k in ("RADIOGRAPH", "X-RAY", "XRAY", "PLAIN FILM")):
        return MODALITY_XRAY
    if "COMPUTED" in token or token.startswith("CT"):
        return MODALITY_CT
    if "ULTRASOUND" in token or "SONOGRAM" in token:
        return MODALITY_ULTRASOUND
    return MODALITY_OTHER


def _normalise_side(value: Any) -> str:
    if not isinstance(value, str):
        return SIDE_NOT_STATED
    token = value.strip().upper()
    if token in SIDES:
        return token
    if token.startswith("L"):
        return SIDE_LEFT
    if token.startswith("R"):
        return SIDE_RIGHT
    if token.startswith("B"):
        return SIDE_BILATERAL
    return SIDE_NOT_STATED


def _parse_study_date(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    token = value.strip()[:10]
    try:
        parsed = datetime.strptime(token, "%Y-%m-%d")
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc)


def _clean_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    collapsed = " ".join(value.split())
    return collapsed[:limit]


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------

def _base_record(uid: str, source: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "inv_id": str(uuid.uuid4()),
        "modality": MODALITY_OTHER,
        "body_part": INV_DEFAULT_BODY_PART,
        "side": SIDE_NOT_STATED,
        "study_date": None,
        "report_findings": "",
        "report_impression": "",
        "source": source,
        "extraction_status": INV_STATUS_FAILED,
        "extraction_model": None,
        "extracted_at": None,
        "raw_extraction": None,
        "verified_by_uid": None,
        "verified_at": None,
        "file_ref": None,
        "original_filename": None,
        "created_at": now,
        "created_by_uid": uid,
    }


def manual_investigation_record(
    uid: str,
    modality: str,
    side: str,
    study_date: Optional[datetime],
    report_findings: str,
    report_impression: str,
) -> Dict[str, Any]:
    """A record typed directly by the clinician. Agent-visible immediately."""
    record = _base_record(uid, INV_SOURCE_MANUAL_ENTRY)
    record.update({
        "modality": _normalise_modality(modality),
        "side": _normalise_side(side),
        "study_date": study_date,
        "report_findings": _clean_text(report_findings, INV_MAX_FINDINGS_CHARS),
        "report_impression": _clean_text(report_impression, INV_MAX_IMPRESSION_CHARS),
        "extraction_status": INV_STATUS_MANUAL,
    })
    return record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_investigation(
    groq_client,
    uid: str,
    file_bytes: bytes,
    content_type: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract a radiology report into an investigation subdocument.

    Always returns a record. Never raises for model-side problems - failure
    is expressed as extraction_status FAILED or REJECTED so the clinician is
    routed to manual entry rather than shown an error page.

    Raises UploadRejected only for files that should not be sent at all.
    """
    validate_upload(file_bytes, content_type)

    record = _base_record(uid, INV_SOURCE_UPLOADED_REPORT)
    record["original_filename"] = filename

    used_vision = False
    raw_response: Optional[str] = None

    try:
        if content_type == "application/pdf":
            text_layer = _pdf_text_layer(file_bytes)
            if text_layer:
                raw_response = _call_text_model(groq_client, text_layer)
            else:
                pages = _pdf_to_pngs(file_bytes)
                if not pages:
                    record["extraction_status"] = INV_STATUS_FAILED
                    return record
                used_vision = True
                raw_response = _call_vision_model(
                    groq_client, [(page, "image/png") for page in pages]
                )
        else:
            used_vision = True
            raw_response = _call_vision_model(
                groq_client, [(file_bytes, content_type)]
            )
    except Exception:
        logger.exception("Investigation extraction call failed")
        record["extraction_status"] = INV_STATUS_FAILED
        return record

    record["extraction_model"] = INV_EXTRACTION_MODEL if used_vision else TEXT_MODEL
    record["extracted_at"] = datetime.now(timezone.utc)

    parsed = _parse_json(raw_response)
    if parsed is None:
        record["extraction_status"] = INV_STATUS_FAILED
        record["raw_extraction"] = {"unparsed": (raw_response or "")[:2000]}
        return record

    # raw_extraction is written once and never mutated by later edits. The
    # difference between this and the clinician-verified fields is the
    # extraction-accuracy evidence for the evaluation chapter.
    record["raw_extraction"] = parsed

    # Refusal path. The decision is made here, in Python, from a boolean the
    # model returned - the model does not decide what happens next.
    if parsed.get("is_report") is not True:
        record["extraction_status"] = INV_STATUS_REJECTED
        return record

    findings = _clean_text(parsed.get("report_findings"), INV_MAX_FINDINGS_CHARS)
    impression = _clean_text(parsed.get("report_impression"), INV_MAX_IMPRESSION_CHARS)

    # A "report" with no transcribable text is not usable. Treat as rejected
    # rather than presenting an empty record for verification.
    if not findings and not impression:
        record["extraction_status"] = INV_STATUS_REJECTED
        return record

    record.update({
        "modality": _normalise_modality(parsed.get("modality")),
        "body_part": _clean_text(parsed.get("body_part"), 60) or INV_DEFAULT_BODY_PART,
        "side": _normalise_side(parsed.get("side")),
        "study_date": _parse_study_date(parsed.get("study_date")),
        "report_findings": findings,
        "report_impression": impression,
        "extraction_status": INV_STATUS_EXTRACTED,
    })

    return record