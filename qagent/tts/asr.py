import os
import json
import base64
import time
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from functools import wraps
from http import HTTPStatus
from urllib import request as url_request

import dashscope
import pyaudio
from cachetools import LRUCache
from dashscope.audio.asr import (
    Recognition,
    RecognitionCallback,
    RecognitionResult,
    TranslationRecognizerRealtime,
    TranslationRecognizerChat,
    TranslationRecognizerCallback,
    TranscriptionResult,
    TranslationResult,
    Transcription,
)
from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation,
    OmniRealtimeCallback,
    MultiModality,
)
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class ASRException(Exception):
    pass


def asr_cache(maxsize=128):
    cache = LRUCache(maxsize=maxsize)

    def decorator(func):
        @wraps(func)
        def wrapper(self, audio_file: str, use_cache: bool = True, **kwargs):
            if not use_cache:
                return func(self, audio_file, **kwargs)

            cache_key = self._get_cache_key(audio_file)
            if cache_key in cache:
                self.logger.info(f"Cache hit for key: {cache_key[:16]}...")
                return cache[cache_key]

            result = func(self, audio_file, **kwargs)
            cache[cache_key] = result
            self.logger.info(f"Cached result for key: {cache_key[:16]}...")
            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = lambda: {
            "currsize": len(cache),
            "maxsize": cache.maxsize,
        }
        return wrapper

    return decorator


@dataclass
class ClientConfig:
    api_key: Optional[str] = None
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ali_api_key")
        if not self.api_key:
            raise ASRException("API key not found")


@dataclass
class RecognitionConfig:
    model: str = "paraformer-realtime-v2"
    format: str = "pcm"
    sample_rate: int = 16000
    language_hints: Optional[List[str]] = None

    def __post_init__(self):
        if self.language_hints is None:
            self.language_hints = []


@dataclass
class TranslationConfig:
    model: str = "gummy-realtime-v1"
    format: str = "pcm"
    sample_rate: int = 16000
    transcription_enabled: bool = True
    translation_enabled: bool = True
    translation_target_languages: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class BatchTranscriptionConfig:
    model: str = "paraformer-v2"
    language_hints: Optional[List[str]] = None

    def __post_init__(self):
        if self.language_hints is None:
            self.language_hints = []


@dataclass
class Qwen3ASRConfig:
    model: str = "qwen3-asr-flash-realtime"
    language: str = "zh"
    sample_rate: int = 16000
    input_audio_format: str = "pcm"
    corpus_text: str = ""
    url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


class RealtimeRecognitionCallback(RecognitionCallback):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._player = None
        self._stream = None
        self.results = []

    def on_open(self) -> None:
        logger.info("Recognition opened")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True
        )

    def on_close(self) -> None:
        logger.info("Recognition closed")
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._player:
            self._player.terminate()
        self._stream = None
        self._player = None

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        self.results.append(sentence)
        logger.info(f"Recognized: {sentence}")

    def get_stream(self):
        return self._stream

    def get_results(self) -> List[str]:
        return self.results


class RealtimeTranslationCallback(TranslationRecognizerCallback):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._player = None
        self._stream = None
        self.transcriptions = []
        self.translations = {}

    def on_open(self) -> None:
        logger.info("Translation recognizer opened")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True
        )

    def on_close(self) -> None:
        logger.info("Translation recognizer closed")
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._player:
            self._player.terminate()
        self._stream = None
        self._player = None

    def on_event(
        self,
        request_id,
        transcription_result: TranscriptionResult,
        translation_result: TranslationResult,
        usage,
    ) -> None:
        logger.info(f"Request ID: {request_id}, Usage: {usage}")

        if transcription_result is not None:
            text = transcription_result.text
            self.transcriptions.append(text)
            logger.info(f"Transcription: {text}")

        if translation_result is not None:
            for lang in translation_result.get_language_list():
                trans = translation_result.get_translation(lang)
                if lang not in self.translations:
                    self.translations[lang] = []
                self.translations[lang].append(trans.text)
                logger.info(f"Translation [{lang}]: {trans.text}")

    def get_stream(self):
        return self._stream

    def get_transcriptions(self) -> List[str]:
        return self.transcriptions

    def get_translations(self) -> Dict[str, List[str]]:
        return self.translations


class ParaformerASR:
    def __init__(self, client_cfg: ClientConfig, recognition_cfg: RecognitionConfig):
        self.client_cfg = client_cfg
        self.recognition_cfg = recognition_cfg
        dashscope.api_key = client_cfg.api_key
        self.logger = logging.getLogger(__name__)
        self._create_retry_decorator()
        self._last_request_id = None
        self._first_package_delay = None
        self._last_package_delay = None

    def _create_retry_decorator(self):
        self.retry_decorator = retry(
            retry=retry_if_exception_type((Exception,)),
            stop=stop_after_attempt(self.client_cfg.max_retries),
            wait=wait_exponential(
                min=self.client_cfg.retry_min_wait, max=self.client_cfg.retry_max_wait
            ),
            reraise=True,
        )

    def _get_cache_key(self, audio_file: str) -> str:
        file_hash = hashlib.sha256(Path(audio_file).read_bytes()).hexdigest()
        key_data = {
            "file_hash": file_hash,
            "model": self.recognition_cfg.model,
            "format": self.recognition_cfg.format,
            "sample_rate": self.recognition_cfg.sample_rate,
            "language_hints": self.recognition_cfg.language_hints,
        }
        json_str = json.dumps(key_data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @asr_cache(maxsize=128)
    def recognize_file(self, audio_file: str) -> Union[str, List[Dict[str, Any]]]:
        if not Path(audio_file).exists():
            raise ASRException(f"Audio file not found: {audio_file}")

        @self.retry_decorator
        def _do_recognize():
            recognition = Recognition(
                model=self.recognition_cfg.model,
                format=self.recognition_cfg.format,
                sample_rate=self.recognition_cfg.sample_rate,
                language_hints=self.recognition_cfg.language_hints,
                callback=None,
            )
            result = recognition.call(audio_file)

            if result.status_code != HTTPStatus.OK:
                raise ASRException(f"Recognition failed: {result.message}")

            self._last_request_id = recognition.get_last_request_id()
            self._first_package_delay = recognition.get_first_package_delay()
            self._last_package_delay = recognition.get_last_package_delay()

            sentences = result.get_sentence()
            if isinstance(sentences, list):
                text = " ".join([s.get("text", "") for s in sentences if "text" in s])
                self.logger.info(f"Recognition result: {text}")
                return sentences
            else:
                self.logger.info(f"Recognition result: {sentences}")
                return sentences

        return _do_recognize()

    def recognize_stream(self, duration: int = 10) -> List[str]:
        callback = RealtimeRecognitionCallback(sample_rate=self.recognition_cfg.sample_rate)

        recognition = Recognition(
            model=self.recognition_cfg.model,
            format=self.recognition_cfg.format,
            sample_rate=self.recognition_cfg.sample_rate,
            callback=callback,
        )
        recognition.start()

        self.logger.info(f"Recording for {duration} seconds...")
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                stream = callback.get_stream()
                if stream:
                    data = stream.read(3200, exception_on_overflow=False)
                    recognition.send_audio_frame(data)
                else:
                    break
        finally:
            recognition.stop()

        self._last_request_id = recognition.get_last_request_id()
        self._first_package_delay = recognition.get_first_package_delay()
        self._last_package_delay = recognition.get_last_package_delay()

        return callback.get_results()

    def recognize_file_stream(self, audio_file: str) -> str:
        if not Path(audio_file).exists():
            raise ASRException(f"Audio file not found: {audio_file}")

        class FileCallback(RecognitionCallback):
            def __init__(self):
                self.text = ""

            def on_open(self) -> None:
                logger.info("File stream recognition started")

            def on_close(self) -> None:
                logger.info("File stream recognition closed")

            def on_error(self, result: RecognitionResult) -> None:
                raise ASRException(f"Recognition error: {result.message}")

            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                if "text" in sentence and RecognitionResult.is_sentence_end(sentence):
                    self.text += sentence["text"]

        callback = FileCallback()
        recognition = Recognition(
            model=self.recognition_cfg.model,
            format=self.recognition_cfg.format,
            sample_rate=self.recognition_cfg.sample_rate,
            callback=callback,
        )

        recognition.start()

        try:
            with open(audio_file, "rb") as f:
                while chunk := f.read(3200):
                    recognition.send_audio_frame(chunk)
                    time.sleep(0.01)
        finally:
            recognition.stop()

        self._last_request_id = recognition.get_last_request_id()
        self._first_package_delay = recognition.get_first_package_delay()
        self._last_package_delay = recognition.get_last_package_delay()

        return callback.text

    def get_last_request_id(self) -> Optional[str]:
        return self._last_request_id

    def get_first_package_delay(self) -> Optional[int]:
        return self._first_package_delay

    def get_last_package_delay(self) -> Optional[int]:
        return self._last_package_delay

    def cache_info(self):
        return self.recognize_file.cache_info()

    def cache_clear(self):
        self.recognize_file.cache_clear()


class GummyASR:
    def __init__(
        self, client_cfg: ClientConfig, translation_cfg: TranslationConfig, is_chat: bool = False
    ):
        self.client_cfg = client_cfg
        self.translation_cfg = translation_cfg
        self.is_chat = is_chat
        dashscope.api_key = client_cfg.api_key
        self.logger = logging.getLogger(__name__)
        self._last_request_id = None
        self._first_package_delay = None
        self._last_package_delay = None

    def recognize_file(self, audio_file: str) -> Dict[str, Any]:
        if not Path(audio_file).exists():
            raise ASRException(f"Audio file not found: {audio_file}")

        class FileTranslationCallback(TranslationRecognizerCallback):
            def __init__(self):
                self.transcriptions = []
                self.translations = {}

            def on_open(self) -> None:
                logger.info("File translation started")

            def on_close(self) -> None:
                logger.info("File translation closed")

            def on_error(self, message) -> None:
                raise ASRException(f"Translation error: {message}")

            def on_event(
                self,
                request_id,
                transcription_result: TranscriptionResult,
                translation_result: TranslationResult,
                usage,
            ) -> None:
                if transcription_result is not None and transcription_result.is_sentence_end:
                    self.transcriptions.append(transcription_result.text)

                if translation_result is not None:
                    for lang in translation_result.get_language_list():
                        trans = translation_result.get_translation(lang)
                        if trans.is_sentence_end:
                            if lang not in self.translations:
                                self.translations[lang] = []
                            self.translations[lang].append(trans.text)

        callback = FileTranslationCallback()

        translator = TranslationRecognizerRealtime(
            model=self.translation_cfg.model,
            format=self.translation_cfg.format,
            sample_rate=self.translation_cfg.sample_rate,
            transcription_enabled=self.translation_cfg.transcription_enabled,
            translation_enabled=self.translation_cfg.translation_enabled,
            translation_target_languages=self.translation_cfg.translation_target_languages,
            callback=callback,
        )

        translator.start()

        try:
            with open(audio_file, "rb") as f:
                while chunk := f.read(3200):
                    translator.send_audio_frame(chunk)
                    time.sleep(0.01)
        finally:
            translator.stop()

        self._last_request_id = translator.get_last_request_id()
        self._first_package_delay = translator.get_first_package_delay()
        self._last_package_delay = translator.get_last_package_delay()

        return {"transcriptions": callback.transcriptions, "translations": callback.translations}

    def recognize_stream(self, duration: int = 10) -> Dict[str, Any]:
        callback = RealtimeTranslationCallback(sample_rate=self.translation_cfg.sample_rate)

        if self.is_chat:
            recognizer = TranslationRecognizerChat(
                model=self.translation_cfg.model,
                format=self.translation_cfg.format,
                sample_rate=self.translation_cfg.sample_rate,
                transcription_enabled=self.translation_cfg.transcription_enabled,
                translation_enabled=self.translation_cfg.translation_enabled,
                translation_target_languages=self.translation_cfg.translation_target_languages,
                callback=callback,
            )
        else:
            recognizer = TranslationRecognizerRealtime(
                model=self.translation_cfg.model,
                format=self.translation_cfg.format,
                sample_rate=self.translation_cfg.sample_rate,
                transcription_enabled=self.translation_cfg.transcription_enabled,
                translation_enabled=self.translation_cfg.translation_enabled,
                translation_target_languages=self.translation_cfg.translation_target_languages,
                callback=callback,
            )

        recognizer.start()

        self.logger.info(f"Recording for {duration} seconds...")
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                stream = callback.get_stream()
                if stream:
                    data = stream.read(3200, exception_on_overflow=False)
                    if self.is_chat:
                        if not recognizer.send_audio_frame(data):
                            self.logger.info("Sentence end detected")
                            break
                    else:
                        recognizer.send_audio_frame(data)
                else:
                    break
        finally:
            recognizer.stop()

        self._last_request_id = recognizer.get_last_request_id()
        self._first_package_delay = recognizer.get_first_package_delay()
        self._last_package_delay = recognizer.get_last_package_delay()

        return {
            "transcriptions": callback.get_transcriptions(),
            "translations": callback.get_translations(),
        }

    def get_last_request_id(self) -> Optional[str]:
        return self._last_request_id

    def get_first_package_delay(self) -> Optional[int]:
        return self._first_package_delay

    def get_last_package_delay(self) -> Optional[int]:
        return self._last_package_delay


class BatchTranscriptionASR:
    def __init__(self, client_cfg: ClientConfig, batch_cfg: BatchTranscriptionConfig):
        self.client_cfg = client_cfg
        self.batch_cfg = batch_cfg
        dashscope.api_key = client_cfg.api_key
        self.logger = logging.getLogger(__name__)
        self._last_task_id = None

    def transcribe_urls(self, file_urls: List[str], wait_completion: bool = True) -> Dict[str, Any]:
        if not file_urls:
            raise ASRException("file_urls cannot be empty")

        task_response = Transcription.async_call(
            model=self.batch_cfg.model,
            language_hints=self.batch_cfg.language_hints if self.batch_cfg.language_hints else None,
            file_urls=file_urls,
        )

        self._last_task_id = task_response.output.task_id
        self.logger.info(f"Task submitted: {self._last_task_id}")

        if not wait_completion:
            return {"task_id": self._last_task_id, "status": "submitted"}

        result = Transcription.wait(task=self._last_task_id)

        if result.status_code == HTTPStatus.OK:
            transcriptions = []
            for item in result.output.get("results", []):
                if item.get("subtask_status") == "SUCCEEDED":
                    url = item.get("transcription_url")
                    if url:
                        try:
                            response = url_request.urlopen(url)
                            data = json.loads(response.read().decode("utf8"))
                            transcriptions.append(data)
                        except Exception as e:
                            self.logger.error(f"Failed to fetch transcription: {e}")
                            transcriptions.append({"error": str(e)})
                else:
                    transcriptions.append({"error": "Transcription failed", "details": item})

            return {
                "task_id": self._last_task_id,
                "status": "completed",
                "transcriptions": transcriptions,
            }
        else:
            raise ASRException(f"Batch transcription failed: {result.output.message}")

    def get_task_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        tid = task_id or self._last_task_id
        if not tid:
            raise ASRException("No task_id available")

        result = Transcription.fetch(task=tid)

        if result.status_code == HTTPStatus.OK:
            return {
                "task_id": tid,
                "status": result.output.get("task_status"),
                "results": result.output.get("results", []),
            }
        else:
            raise ASRException(f"Failed to fetch task status: {result.output.message}")


class Qwen3ASR:
    def __init__(self, client_cfg: ClientConfig, qwen3_cfg: Qwen3ASRConfig):
        self.client_cfg = client_cfg
        self.qwen3_cfg = qwen3_cfg
        dashscope.api_key = client_cfg.api_key
        self.logger = logging.getLogger(__name__)
        self.conversation = None
        self._session_id = None
        self._response_id = None
        self._first_text_delay = None

    def recognize_file(self, audio_file: str) -> List[str]:
        if not Path(audio_file).exists():
            raise ASRException(f"Audio file not found: {audio_file}")

        transcriptions = []

        class Qwen3Callback(OmniRealtimeCallback):
            def on_open(self):
                logger.info("Qwen3 ASR connection opened")

            def on_close(self, code, msg):
                logger.info(f"Qwen3 ASR connection closed: {code}, {msg}")

            def on_event(self, response):
                event_type = response.get("type")
                if event_type == "session.created":
                    logger.info(f"Session created: {response['session']['id']}")
                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcriptions.append(response.get("transcript", ""))
                elif event_type == "conversation.item.input_audio_transcription.text":
                    logger.info(f"Stash: {response.get('stash', '')}")

        callback = Qwen3Callback()

        self.conversation = OmniRealtimeConversation(
            model=self.qwen3_cfg.model, url=self.qwen3_cfg.url, callback=callback
        )

        self.conversation.connect()

        transcription_params = TranscriptionParams(
            language=self.qwen3_cfg.language,
            sample_rate=self.qwen3_cfg.sample_rate,
            input_audio_format=self.qwen3_cfg.input_audio_format,
            corpus_text=self.qwen3_cfg.corpus_text,
        )

        self.conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params,
        )

        try:
            with open(audio_file, "rb") as f:
                while chunk := f.read(3200):
                    audio_b64 = base64.b64encode(chunk).decode("ascii")
                    self.conversation.append_audio(audio_b64)
                    time.sleep(0.01)

            silence_data = bytes(1024)
            for _ in range(30):
                audio_b64 = base64.b64encode(silence_data).decode("ascii")
                self.conversation.append_audio(audio_b64)
                time.sleep(0.01)

            time.sleep(2)
        finally:
            self._session_id = self.conversation.get_session_id()
            self._response_id = self.conversation.get_last_response_id()
            self._first_text_delay = self.conversation.get_last_first_text_delay()
            self.conversation.close()

        return transcriptions

    def recognize_stream(self, duration: int = 10) -> List[str]:
        transcriptions = []

        class Qwen3MicCallback(OmniRealtimeCallback):
            def __init__(self):
                self._player = None
                self._stream = None

            def on_open(self):
                logger.info("Qwen3 ASR microphone opened")
                self._player = pyaudio.PyAudio()
                self._stream = self._player.open(
                    format=pyaudio.paInt16, channels=1, rate=16000, input=True
                )

            def on_close(self, code, msg):
                logger.info(f"Qwen3 ASR closed: {code}, {msg}")
                if self._stream:
                    self._stream.stop_stream()
                    self._stream.close()
                if self._player:
                    self._player.terminate()

            def on_event(self, response):
                event_type = response.get("type")
                if event_type == "conversation.item.input_audio_transcription.completed":
                    transcriptions.append(response.get("transcript", ""))
                elif event_type == "input_audio_buffer.speech_started":
                    logger.info("Speech started")
                elif event_type == "input_audio_buffer.speech_stopped":
                    logger.info("Speech stopped")

            def get_stream(self):
                return self._stream

        callback = Qwen3MicCallback()

        self.conversation = OmniRealtimeConversation(
            model=self.qwen3_cfg.model, url=self.qwen3_cfg.url, callback=callback
        )

        self.conversation.connect()

        transcription_params = TranscriptionParams(
            language=self.qwen3_cfg.language,
            sample_rate=16000,
            input_audio_format="pcm",
            corpus_text=self.qwen3_cfg.corpus_text,
        )

        self.conversation.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=transcription_params,
        )

        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                stream = callback.get_stream()
                if stream:
                    audio_data = stream.read(3200, exception_on_overflow=False)
                    audio_b64 = base64.b64encode(audio_data).decode("ascii")
                    self.conversation.append_audio(audio_b64)
                else:
                    break
        finally:
            self._session_id = self.conversation.get_session_id()
            self._response_id = self.conversation.get_last_response_id()
            self._first_text_delay = self.conversation.get_last_first_text_delay()
            self.conversation.close()

        return transcriptions

    def get_session_id(self) -> Optional[str]:
        return self._session_id

    def get_last_response_id(self) -> Optional[str]:
        return self._response_id

    def get_last_first_text_delay(self) -> Optional[int]:
        return self._first_text_delay


class ASRFactory:
    @staticmethod
    def create_paraformer(
        api_key: Optional[str] = None,
        model: str = "paraformer-realtime-v2",
        format: str = "pcm",
        sample_rate: int = 16000,
        language_hints: Optional[List[str]] = None,
    ) -> ParaformerASR:
        client_cfg = ClientConfig(api_key=api_key)
        recognition_cfg = RecognitionConfig(
            model=model, format=format, sample_rate=sample_rate, language_hints=language_hints
        )
        return ParaformerASR(client_cfg, recognition_cfg)

    @staticmethod
    def create_fun_asr(
        api_key: Optional[str] = None,
        format: str = "pcm",
        sample_rate: int = 16000,
        language_hints: Optional[List[str]] = None,
    ) -> ParaformerASR:
        client_cfg = ClientConfig(api_key=api_key)
        recognition_cfg = RecognitionConfig(
            model="fun-asr-realtime",
            format=format,
            sample_rate=sample_rate,
            language_hints=language_hints,
        )
        return ParaformerASR(client_cfg, recognition_cfg)

    @staticmethod
    def create_gummy(
        api_key: Optional[str] = None,
        model: str = "gummy-realtime-v1",
        is_chat: bool = False,
        translation_languages: Optional[List[str]] = None,
    ) -> GummyASR:
        client_cfg = ClientConfig(api_key=api_key)
        translation_cfg = TranslationConfig(
            model=model if not is_chat else "gummy-chat-v1",
            translation_target_languages=translation_languages or ["en"],
        )
        return GummyASR(client_cfg, translation_cfg, is_chat=is_chat)

    @staticmethod
    def create_batch_transcription(
        api_key: Optional[str] = None,
        model: str = "paraformer-v2",
        language_hints: Optional[List[str]] = None,
    ) -> BatchTranscriptionASR:
        client_cfg = ClientConfig(api_key=api_key)
        batch_cfg = BatchTranscriptionConfig(model=model, language_hints=language_hints)
        return BatchTranscriptionASR(client_cfg, batch_cfg)

    @staticmethod
    def create_qwen3_asr(
        api_key: Optional[str] = None,
        model: str = "qwen3-asr-flash-realtime",
        language: str = "zh",
        sample_rate: int = 16000,
        corpus_text: str = "",
        url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    ) -> Qwen3ASR:
        client_cfg = ClientConfig(api_key=api_key)
        qwen3_cfg = Qwen3ASRConfig(
            model=model,
            language=language,
            sample_rate=sample_rate,
            corpus_text=corpus_text,
            url=url,
        )
        return Qwen3ASR(client_cfg, qwen3_cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    os.makedirs("test_outputs", exist_ok=True)

    print("\n" + "=" * 80)
    print("ASR Quick Test - Basic Functionality")
    print("For comprehensive tests, run: py test_asr_complete.py")
    print("=" * 80)

    import requests

    print("\n" + "=" * 80)
    print("Test 1: Paraformer File Recognition + Cache")
    print("=" * 80)

    audio_url = "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav"
    test_audio = "test_outputs/test_audio.wav"

    print("1.1 Download test audio")
    r = requests.get(audio_url)
    with open(test_audio, "wb") as f:
        f.write(r.content)
    print(f"    Downloaded: {test_audio}")

    paraformer = ASRFactory.create_paraformer(language_hints=["zh", "en"])

    print("1.2 First recognition")
    result1 = paraformer.recognize_file(test_audio, use_cache=True)
    print(f"    Result: {result1}")
    print(f"    Cache: {paraformer.cache_info()}")

    print("1.3 Second recognition (cache hit)")
    result2 = paraformer.recognize_file(test_audio, use_cache=True)
    print(f"    Cache hit: {result1 == result2}")
    print(f"    Cache: {paraformer.cache_info()}")

    print("\n" + "=" * 80)
    print("Test 2: Paraformer File Stream")
    print("=" * 80)

    result_stream = paraformer.recognize_file_stream(test_audio)
    print(f"    Stream result: {result_stream}")
    print(f"    Metrics - Request: {paraformer.get_last_request_id()}")
    print(f"              First delay: {paraformer.get_first_package_delay()}ms")
    print(f"              Last delay: {paraformer.get_last_package_delay()}ms")

    print("\n" + "=" * 80)
    print("Test 3: Gummy File Translation")
    print("=" * 80)

    gummy = ASRFactory.create_gummy(translation_languages=["en"])
    result_trans = gummy.recognize_file(test_audio)
    print(f"    Transcriptions: {result_trans['transcriptions']}")
    print(f"    Translations: {result_trans['translations']}")
    print(f"    Metrics - Request: {gummy.get_last_request_id()}")

    print("\n" + "=" * 80)
    print("Test 4: Batch Transcription (Async)")
    print("=" * 80)

    batch_asr = ASRFactory.create_batch_transcription()

    print("4.1 Submit job")
    file_urls = [
        "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world_female2.wav"
    ]
    result_batch = batch_asr.transcribe_urls(file_urls, wait_completion=True)
    print(f"    Task ID: {result_batch['task_id']}")
    print(f"    Status: {result_batch['status']}")
    print(f"    Results: {len(result_batch.get('transcriptions', []))} files")

    print("\n" + "=" * 80)
    print("Test 5: Qwen3-ASR File Recognition")
    print("=" * 80)

    qwen3 = ASRFactory.create_qwen3_asr(language="zh")
    transcriptions = qwen3.recognize_file(test_audio)
    print(f"    Transcriptions: {transcriptions}")
    print(f"    Session: {qwen3.get_session_id()}")
    print(f"    Response: {qwen3.get_last_response_id()}")

    print("\n" + "=" * 80)
    print("Quick Tests Completed!")
    print("Run 'py test_asr_complete.py' for full test suite")
    print("=" * 80)
