import os
import tempfile
import threading
import requests
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from silero_vad import load_silero_vad, get_speech_timestamps


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class InsufficientSpeechError(ValueError):
    """Поднимается, когда после VAD остаётся слишком мало речи для надёжного эмбеддинга."""


# приоритет decision'ов для выбора лучшего канала в identify_auto
_DECISION_PRIORITY = {
    "match": 4,
    "low_confidence": 3,
    "unknown": 2,
    "no_employees": 1,
    "no_speech": 0,
}


class VoiceIDService:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/speechbrain"),
            run_opts={"device": self.device},
        )
        self.model.eval()
        self.vad = load_silero_vad()

        self.threshold = float(os.getenv("MATCH_THRESHOLD", "0.40"))
        self.margin_threshold = float(os.getenv("MARGIN_THRESHOLD", "0.15"))
        self.min_speech_duration = float(os.getenv("MIN_SPEECH_DURATION", "1.5"))
        self.vad_min_speech_ms = int(os.getenv("VAD_MIN_SPEECH_MS", "250"))
        self.vad_min_silence_ms = int(os.getenv("VAD_MIN_SILENCE_MS", "300"))
        self.vad_speech_pad_ms = int(os.getenv("VAD_SPEECH_PAD_MS", "100"))

        self.lock = threading.Lock()

    # ====== I/O ======

    def _download(self, url: str) -> str:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        ext = ".wav"
        if "." in url.split("/")[-1]:
            ext = "." + url.split("/")[-1].split(".")[-1].split("?")[0]
        if ext not in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]:
            ext = ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(resp.content)
        tmp.close()
        return tmp.name

    # ====== preprocessing ======

    def _to_mono_16k(self, signal: torch.Tensor, sr: int):
        if sr != 16000:
            signal = torchaudio.transforms.Resample(sr, 16000)(signal)
            sr = 16000
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal, sr

    def _channel_16k(self, signal: torch.Tensor, sr: int, channel: int):
        if signal.shape[0] != 2:
            raise ValueError("Ожидается stereo файл (2 канала)")
        ch_signal = signal[channel : channel + 1]
        if sr != 16000:
            ch_signal = torchaudio.transforms.Resample(sr, 16000)(ch_signal)
            sr = 16000
        return ch_signal, sr

    def _apply_vad(self, signal: torch.Tensor, sr: int):
        """signal: моно [1, T] на 16 кГц.
        Возвращает (clean [1, T'], voiced_seconds, n_segments)."""
        x = signal.squeeze(0)
        segs = get_speech_timestamps(
            x,
            self.vad,
            sampling_rate=sr,
            min_speech_duration_ms=self.vad_min_speech_ms,
            min_silence_duration_ms=self.vad_min_silence_ms,
            speech_pad_ms=self.vad_speech_pad_ms,
        )
        if not segs:
            return signal[:, :0], 0.0, 0
        voiced = torch.cat([x[s["start"] : s["end"]] for s in segs]).unsqueeze(0)
        return voiced, voiced.shape[1] / sr, len(segs)

    def _embed(self, signal: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.model.encode_batch(signal)
        return emb.squeeze().cpu().numpy()

    def _vad_or_raise(self, signal: torch.Tensor, sr: int):
        clean, voiced_sec, n_segs = self._apply_vad(signal, sr)
        if voiced_sec < self.min_speech_duration:
            raise InsufficientSpeechError(
                f"Слишком мало речи после VAD: {voiced_sec:.2f}s "
                f"(минимум {self.min_speech_duration}s). "
                f"Семпл должен содержать больше чистой речи без длинных пауз."
            )
        return clean, voiced_sec, n_segs

    # ====== EXTRACT ======

    def extract_from_url(self, url: str) -> np.ndarray:
        """Извлекает вектор из URL семпла. Возвращает numpy array 192D."""
        with self.lock:
            path = self._download(url)
            try:
                signal, sr = torchaudio.load(path)
                signal, sr = self._to_mono_16k(signal, sr)
                clean, _, _ = self._vad_or_raise(signal, sr)
                return self._embed(clean)
            finally:
                os.unlink(path)

    def extract_from_file(self, file_path: str) -> np.ndarray:
        with self.lock:
            signal, sr = torchaudio.load(file_path)
            signal, sr = self._to_mono_16k(signal, sr)
            clean, _, _ = self._vad_or_raise(signal, sr)
            return self._embed(clean)

    def extract_channel_from_url(self, url: str, channel: int) -> np.ndarray:
        if channel not in (0, 1):
            raise ValueError("channel должен быть 0 или 1")
        with self.lock:
            path = self._download(url)
            try:
                signal, sr = torchaudio.load(path)
                ch_signal, sr = self._channel_16k(signal, sr, channel)
                clean, _, _ = self._vad_or_raise(ch_signal, sr)
                return self._embed(clean)
            finally:
                os.unlink(path)

    def extract_channel_from_file(self, file_path: str, channel: int) -> np.ndarray:
        if channel not in (0, 1):
            raise ValueError("channel должен быть 0 или 1")
        with self.lock:
            signal, sr = torchaudio.load(file_path)
            ch_signal, sr = self._channel_16k(signal, sr, channel)
            clean, _, _ = self._vad_or_raise(ch_signal, sr)
            return self._embed(clean)

    # ====== IDENTIFY ======

    def _decide(self, top1: float, margin: float, voiced_sec: float):
        if (
            top1 >= self.threshold
            and margin >= self.margin_threshold
            and voiced_sec >= self.min_speech_duration
        ):
            return "match", True
        if top1 >= self.threshold * 0.8 and margin >= self.margin_threshold * 0.5:
            return "low_confidence", False
        return "unknown", False

    def _score(self, call_emb: np.ndarray, employee_vectors: list):
        scored = []
        for emp in employee_vectors:
            score = _cosine_similarity(call_emb, emp["embedding"])
            scored.append(
                {
                    "employee_id": emp["id"],
                    "employee_name": emp["name"],
                    "score": round(score, 4),
                    "_emp": emp,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        top1 = scored[0]["score"]
        top2 = scored[1]["score"] if len(scored) > 1 else 0.0
        margin = top1 - top2
        best_emp = scored[0]["_emp"]
        top_scores = [{k: v for k, v in s.items() if k != "_emp"} for s in scored[:5]]
        return best_emp, top1, margin, top_scores

    def _no_speech_result(self, channel: int, voiced_sec: float, n_segs: int):
        return {
            "identified_employee_id": None,
            "identified_employee_name": None,
            "confidence": 0.0,
            "is_match": False,
            "threshold": self.threshold,
            "employee_channel": channel,
            "top_scores": [],
            # extra (опц., Bubble может игнорировать):
            "margin": 0.0,
            "decision": "no_speech",
            "speech_duration_sec": round(voiced_sec, 2),
            "vad_segments_count": n_segs,
        }

    def _no_employees_result(self, channel: int, voiced_sec: float, n_segs: int):
        return {
            "identified_employee_id": None,
            "identified_employee_name": None,
            "confidence": 0.0,
            "is_match": False,
            "threshold": self.threshold,
            "employee_channel": channel,
            "top_scores": [],
            "margin": 0.0,
            "decision": "no_employees",
            "speech_duration_sec": round(voiced_sec, 2),
            "vad_segments_count": n_segs,
        }

    def _build_result(
        self,
        call_emb: np.ndarray,
        employee_vectors: list,
        channel: int,
        voiced_sec: float,
        n_segs: int,
    ):
        if not employee_vectors:
            return self._no_employees_result(channel, voiced_sec, n_segs)
        best_emp, top1, margin, top_scores = self._score(call_emb, employee_vectors)
        decision, is_match = self._decide(top1, margin, voiced_sec)
        return {
            "identified_employee_id": best_emp["id"] if is_match else None,
            "identified_employee_name": best_emp["name"] if is_match else None,
            "confidence": round(top1, 4),
            "is_match": is_match,
            "threshold": self.threshold,
            "employee_channel": channel,
            "top_scores": top_scores,
            # extra (опц., Bubble может игнорировать):
            "margin": round(margin, 4),
            "decision": decision,
            "speech_duration_sec": round(voiced_sec, 2),
            "vad_segments_count": n_segs,
        }

    def identify(self, call_url: str, employee_channel: int, employee_vectors: list) -> dict:
        with self.lock:
            call_path = self._download(call_url)
            try:
                signal, sr = torchaudio.load(call_path)
                ch_signal, sr = self._channel_16k(signal, sr, employee_channel)
                clean, voiced_sec, n_segs = self._apply_vad(ch_signal, sr)
                if voiced_sec < self.min_speech_duration:
                    return self._no_speech_result(employee_channel, voiced_sec, n_segs)
                call_emb = self._embed(clean)
            finally:
                os.unlink(call_path)
            return self._build_result(
                call_emb, employee_vectors, employee_channel, voiced_sec, n_segs
            )

    def identify_auto(self, call_url: str, employee_vectors: list) -> dict:
        """Проверяет оба канала, выбирает лучший по приоритету decision → confidence."""
        with self.lock:
            call_path = self._download(call_url)
            try:
                signal, sr = torchaudio.load(call_path)
                if signal.shape[0] != 2:
                    raise ValueError("Ожидается stereo файл (2 канала)")
                if sr != 16000:
                    signal = torchaudio.transforms.Resample(sr, 16000)(signal)
                    sr = 16000

                channel_results = []
                for ch in (0, 1):
                    ch_signal = signal[ch : ch + 1]
                    clean, voiced_sec, n_segs = self._apply_vad(ch_signal, sr)
                    if voiced_sec < self.min_speech_duration:
                        channel_results.append(
                            self._no_speech_result(ch, voiced_sec, n_segs)
                        )
                        continue
                    call_emb = self._embed(clean)
                    channel_results.append(
                        self._build_result(
                            call_emb, employee_vectors, ch, voiced_sec, n_segs
                        )
                    )

                channel_results.sort(
                    key=lambda r: (
                        _DECISION_PRIORITY.get(r.get("decision", "unknown"), 0),
                        r["confidence"],
                    ),
                    reverse=True,
                )
                return channel_results[0]
            finally:
                os.unlink(call_path)


voice_service = VoiceIDService()
