import os
import tempfile
import threading
import requests
import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class VoiceIDService:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/speechbrain"),
            run_opts={"device": self.device}
        )
        self.model.eval()
        self.threshold = float(os.getenv("MATCH_THRESHOLD", "0.80"))
        self.lock = threading.Lock()

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

    def extract_from_url(self, url: str) -> np.ndarray:
        """Извлекает вектор из URL семпла. Возвращает numpy array 192D."""
        with self.lock:
            path = self._download(url)
            try:
                signal, sr = torchaudio.load(path)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    signal = resampler(signal)
                if signal.shape[0] > 1:
                    signal = torch.mean(signal, dim=0, keepdim=True)
                with torch.no_grad():
                    emb = self.model.encode_batch(signal)
                return emb.squeeze().cpu().numpy()
            finally:
                os.unlink(path)

    def extract_from_file(self, file_path: str) -> np.ndarray:
        with self.lock:
            signal, sr = torchaudio.load(file_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                signal = resampler(signal)
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
            with torch.no_grad():
                emb = self.model.encode_batch(signal)
            return emb.squeeze().cpu().numpy()

    def extract_channel_from_url(self, url: str, channel: int) -> np.ndarray:
        """Извлекает вектор из указанного канала (0 или 1) стерео-записи по URL."""
        if channel not in (0, 1):
            raise ValueError("channel должен быть 0 или 1")
        with self.lock:
            path = self._download(url)
            try:
                signal, sr = torchaudio.load(path)
                if signal.shape[0] != 2:
                    raise ValueError("Ожидается stereo файл (2 канала)")
                ch_signal = signal[channel:channel + 1]
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    ch_signal = resampler(ch_signal)
                with torch.no_grad():
                    emb = self.model.encode_batch(ch_signal)
                return emb.squeeze().cpu().numpy()
            finally:
                os.unlink(path)

    def extract_channel_from_file(self, file_path: str, channel: int) -> np.ndarray:
        """Извлекает вектор из указанного канала (0 или 1) стерео-файла."""
        if channel not in (0, 1):
            raise ValueError("channel должен быть 0 или 1")
        with self.lock:
            signal, sr = torchaudio.load(file_path)
            if signal.shape[0] != 2:
                raise ValueError("Ожидается stereo файл (2 канала)")
            ch_signal = signal[channel:channel + 1]
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                ch_signal = resampler(ch_signal)
            with torch.no_grad():
                emb = self.model.encode_batch(ch_signal)
            return emb.squeeze().cpu().numpy()

    def identify(self, call_url: str, employee_channel: int, employee_vectors: list) -> dict:
        """
        employee_vectors: [{"id": "emp1", "name": "Ivan", "embedding": np.ndarray}, ...]
        """
        with self.lock:
            call_path = self._download(call_url)
            try:
                signal, sr = torchaudio.load(call_path)
                if signal.shape[0] != 2:
                    raise ValueError("Ожидается stereo файл (2 канала)")
                emp_signal = signal[employee_channel:employee_channel + 1]
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    emp_signal = resampler(emp_signal)
                with torch.no_grad():
                    call_emb = self.model.encode_batch(emp_signal)
                call_emb = call_emb.squeeze().cpu().numpy()
            finally:
                os.unlink(call_path)

            best_score = -1.0
            best_emp = None
            all_scores = []

            for emp in employee_vectors:
                emb = emp["embedding"]
                score = _cosine_similarity(call_emb, emb)
                all_scores.append({
                    "employee_id": emp["id"],
                    "employee_name": emp["name"],
                    "score": round(score, 4)
                })
                if score > best_score:
                    best_score = score
                    best_emp = emp

            all_scores.sort(key=lambda x: x["score"], reverse=True)
            is_match = best_score >= self.threshold if best_emp else False

            return {
                "identified_employee_id": best_emp["id"] if is_match else None,
                "identified_employee_name": best_emp["name"] if is_match else None,
                "confidence": round(best_score, 4),
                "is_match": is_match,
                "threshold": self.threshold,
                "employee_channel": employee_channel,
                "top_scores": all_scores[:5]
            }

    def identify_auto(self, call_url: str, employee_vectors: list) -> dict:
        """Проверяет оба канала и возвращает результат для лучшего."""
        with self.lock:
            call_path = self._download(call_url)
            try:
                signal, sr = torchaudio.load(call_path)
                if signal.shape[0] != 2:
                    raise ValueError("Ожидается stereo файл (2 канала)")
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    signal = resampler(signal)

                channel_results = []
                for ch in [0, 1]:
                    ch_signal = signal[ch:ch + 1]
                    with torch.no_grad():
                        call_emb = self.model.encode_batch(ch_signal)
                    call_emb = call_emb.squeeze().cpu().numpy()

                    best_score = -1.0
                    best_emp = None
                    all_scores = []

                    for emp in employee_vectors:
                        emb = emp["embedding"]
                        score = _cosine_similarity(call_emb, emb)
                        all_scores.append({
                            "employee_id": emp["id"],
                            "employee_name": emp["name"],
                            "score": round(score, 4)
                        })
                        if score > best_score:
                            best_score = score
                            best_emp = emp

                    all_scores.sort(key=lambda x: x["score"], reverse=True)
                    is_match = best_score >= self.threshold if best_emp else False

                    channel_results.append({
                        "identified_employee_id": best_emp["id"] if is_match else None,
                        "identified_employee_name": best_emp["name"] if is_match else None,
                        "confidence": round(best_score, 4),
                        "is_match": is_match,
                        "threshold": self.threshold,
                        "employee_channel": ch,
                        "top_scores": all_scores[:5]
                    })

                return max(channel_results, key=lambda x: x["confidence"])
            finally:
                os.unlink(call_path)


voice_service = VoiceIDService()
