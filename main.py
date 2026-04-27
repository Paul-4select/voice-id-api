import os
import shutil
from typing import List, Optional, Union
import json
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from services.voice_id import voice_service
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN", "default-token-change-me")
PORT = int(os.getenv("PORT", 8000))

app = FastAPI(title="Voice ID API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logging.info(f"REQUEST {request.method} {request.url.path} BODY: {body.decode('utf-8', errors='replace')[:2000]}")
    response = await call_next(request)
    logging.info(f"RESPONSE {response.status_code}")
    return response

bearer_scheme = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials.credentials


# ====== СХЕМЫ ======

class ExtractRequest(BaseModel):
    sample_url: str = Field(..., description="URL семпла голоса (WAV/MP3)")
    callback_url: Optional[str] = Field(None, description="URL вебхука для дублирования результата")


class ExtractResponse(BaseModel):
    status: str
    embedding: List[float]
    embedding_shape: List[int]


class ExtractChannelRequest(BaseModel):
    audio_url: str = Field(..., description="URL stereo-записи (WAV/MP3)")
    channel: int = Field(..., description="Канал для извлечения: 0 (левый) или 1 (правый)")
    call_id: Optional[str] = Field(None, description="ID звонка для связи с webhook-ответом")
    callback_url: Optional[str] = Field(None, description="URL вебхука для дублирования результата")


class ExtractChannelResponse(BaseModel):
    status: str
    channel: int
    embedding: List[float]
    embedding_string: str
    embedding_shape: List[int]


class EmployeeVector(BaseModel):
    id: str
    name: str
    embedding: List[float]

    @field_validator("embedding", mode="before")
    @classmethod
    def parse_embedding(cls, v):
        if isinstance(v, str):
            return [float(x.strip()) for x in v.split(",")]
        return v


class IdentifyRequest(BaseModel):
    call_id: str = Field(..., description="Уникальный ID звонка")
    call_url: str = Field(..., description="URL записи звонка (stereo WAV/MP3)")
    callback_url: str = Field(..., description="URL куда отправить результат")
    employee_vectors: Union[List[EmployeeVector], str] = Field(..., description="Векторы сотрудников из БД Bubble (массив или JSON-строка)")

    @field_validator("employee_vectors", mode="before")
    @classmethod
    def parse_vectors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    employee_channel: int = Field(2, description="0=left, 1=right, 2=оба канала")


class IdentifyResponse(BaseModel):
    identified_employee_id: Optional[str]
    identified_employee_name: Optional[str]
    confidence: float
    is_match: bool
    threshold: float
    employee_channel: int
    top_scores: List[dict]


class AsyncResponse(BaseModel):
    status: str
    call_id: str
    message: str


# ====== CALLBACK ======

def _send_callback(callback_url: str, payload: dict):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    try:
        resp = requests.post(callback_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"Callback error: {e}")


def _process_identify(call_id: str, call_url: str, callback_url: str, employee_channel: int, employee_vectors: list):
    try:
        if employee_channel == 2:
            result = voice_service.identify_auto(call_url, employee_vectors)
        else:
            result = voice_service.identify(call_url, employee_channel, employee_vectors)
        payload = {"call_id": call_id, "result": result, "processed_at": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        payload = {"call_id": call_id, "error": str(e), "processed_at": datetime.now(timezone.utc).isoformat()}
    _send_callback(callback_url, payload)


def _process_extract(embedding: list, shape: list, callback_url: str):
    payload = {
        "status": "ok",
        "embedding": embedding,
        "embedding_shape": shape,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    _send_callback(callback_url, payload)


def _process_extract_channel(call_id: Optional[str], channel: int, embedding: list, embedding_string: str, shape: list, callback_url: str):
    payload = {
        "status": "ok",
        "call_id": call_id,
        "channel": channel,
        "embedding": embedding,
        "embedding_string": embedding_string,
        "embedding_shape": shape,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    _send_callback(callback_url, payload)


# ====== ЭНДПОИНТЫ ======

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/extract", response_model=ExtractResponse, dependencies=[Depends(verify_token)])
def extract(req: ExtractRequest, background_tasks: BackgroundTasks):
    try:
        emb = voice_service.extract_from_url(req.sample_url)
        response = ExtractResponse(
            status="ok",
            embedding=emb.tolist(),
            embedding_shape=list(emb.shape),
        )
        if req.callback_url:
            background_tasks.add_task(_process_extract, emb.tolist(), list(emb.shape), req.callback_url)
        return response
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения: {str(e)}")


@app.post("/extract/file", dependencies=[Depends(verify_token)])
def extract_file(audio: UploadFile = File(...), callback_url: Optional[str] = Form(None)):
    tmp_path = f"/tmp/extract_{audio.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    try:
        emb = voice_service.extract_from_file(tmp_path)
        response = {
            "status": "ok",
            "embedding": emb.tolist(),
            "embedding_shape": list(emb.shape),
        }
        if callback_url:
            background_tasks.add_task(_process_extract, emb.tolist(), list(emb.shape), callback_url)
        return response
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/extract-channel", response_model=ExtractChannelResponse, dependencies=[Depends(verify_token)])
def extract_channel(req: ExtractChannelRequest, background_tasks: BackgroundTasks):
    if req.channel not in (0, 1):
        raise HTTPException(400, "channel должен быть 0 или 1")
    try:
        emb = voice_service.extract_channel_from_url(req.audio_url, req.channel)
        emb_list = emb.tolist()
        emb_str = ", ".join(str(x) for x in emb_list)
        response = ExtractChannelResponse(
            status="ok",
            channel=req.channel,
            embedding=emb_list,
            embedding_string=emb_str,
            embedding_shape=list(emb.shape),
        )
        if req.callback_url:
            background_tasks.add_task(
                _process_extract_channel, req.call_id, req.channel, emb_list, emb_str, list(emb.shape), req.callback_url
            )
        return response
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения канала: {str(e)}")


@app.post("/extract-channel/file", dependencies=[Depends(verify_token)])
def extract_channel_file(
    request: Request,
    audio: UploadFile = File(...),
    channel: Optional[str] = Form(None),
    call_id: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
):
    # Bubble often passes params as query string when uploading files
    ch_val = channel if channel is not None else request.query_params.get("channel")
    cid_val = call_id if call_id is not None else request.query_params.get("call_id")
    if ch_val is None:
        raise HTTPException(400, "channel is required (pass as query param ?channel=0 or form field)")
    try:
        ch = int(ch_val)
    except ValueError:
        raise HTTPException(400, f"channel must be integer 0 or 1, got: {ch_val}")
    if ch not in (0, 1):
        raise HTTPException(400, "channel must be 0 or 1")

    tmp_path = f"/tmp/extract_ch_{audio.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    try:
        emb = voice_service.extract_channel_from_file(tmp_path, ch)
        emb_list = emb.tolist()
        emb_str = ", ".join(str(x) for x in emb_list)
        response = {
            "status": "ok",
            "channel": ch,
            "embedding": emb_list,
            "embedding_string": emb_str,
            "embedding_shape": list(emb.shape),
        }
        if callback_url:
            background_tasks.add_task(
                _process_extract_channel, cid_val, ch, emb_list, emb_str, list(emb.shape), callback_url
            )
        return response
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Ошибка извлечения канала: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/identify", response_model=AsyncResponse, dependencies=[Depends(verify_token)])
def identify(req: IdentifyRequest, background_tasks: BackgroundTasks):
    if not req.employee_vectors:
        raise HTTPException(400, "employee_vectors пустой")
    if len(req.employee_vectors) > 50:
        raise HTTPException(400, "Максимум 50 сотрудников за раз")

    vectors = []
    for ev in req.employee_vectors:
        vectors.append({
            "id": ev.id,
            "name": ev.name,
            "embedding": np.array(ev.embedding, dtype=np.float32),
        })

    background_tasks.add_task(
        _process_identify,
        call_id=req.call_id,
        call_url=req.call_url,
        callback_url=req.callback_url,
        employee_channel=req.employee_channel,
        employee_vectors=vectors,
    )

    return AsyncResponse(
        status="processing",
        call_id=req.call_id,
        message="Результат будет отправлен на callback_url после обработки",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
