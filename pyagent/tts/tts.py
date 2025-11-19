import os
import base64
import time
import threading
import logging
import queue
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable, List
from enum import Enum

import dashscope
import pyaudio
from dashscope.audio.tts_v2 import (
    AudioFormat,
    ResultCallback,
    SpeechSynthesizer,
)
from dashscope.audio.tts import (
    SpeechSynthesizer as SambertSynthesizer,
    ResultCallback as SambertCallback,
    SpeechSynthesisResult,
)
from dashscope.audio.qwen_tts_realtime import (
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
    AudioFormat as QwenAudioFormat,
)

logger = logging.getLogger(__name__)


class TTSException(Exception):
    pass


class AudioPlayer:
    def __init__(self, sample_rate: int = 24000, chunk_size_ms: int = 100):
        self.sample_rate = sample_rate
        self.chunk_size_bytes = chunk_size_ms * sample_rate * 2 // 1000
        self.pya = pyaudio.PyAudio()
        self.stream = self.pya.open(
            format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True
        )
        self.raw_buffer = queue.Queue()
        self.b64_buffer = queue.Queue()
        self.status_lock = threading.Lock()
        self.status = "playing"
        self.decoder_thread = threading.Thread(target=self._decoder_loop)
        self.player_thread = threading.Thread(target=self._player_loop)
        self.complete_event = None
        self.out_file = None
        self.decoder_thread.start()
        self.player_thread.start()

    def _decoder_loop(self):
        while self.status != "stop":
            data = None
            with contextlib.suppress(queue.Empty):
                data = self.b64_buffer.get(timeout=0.1)
            if data is None:
                continue
            raw = base64.b64decode(data)
            for i in range(0, len(raw), self.chunk_size_bytes):
                chunk = raw[i : i + self.chunk_size_bytes]
                self.raw_buffer.put(chunk)
                if self.out_file:
                    self.out_file.write(chunk)

    def _player_loop(self):
        while self.status != "stop":
            chunk = None
            with contextlib.suppress(queue.Empty):
                chunk = self.raw_buffer.get(timeout=0.1)
            if chunk is None:
                if self.complete_event:
                    self.complete_event.set()
                continue
            self.stream.write(chunk)

    def add_data(self, b64_data: str):
        self.b64_buffer.put(b64_data)

    def wait_complete(self):
        self.complete_event = threading.Event()
        self.complete_event.wait()
        self.complete_event = None

    def set_save_file(self, filepath: str):
        self.out_file = open(filepath, "wb")

    def shutdown(self):
        self.status = "stop"
        self.decoder_thread.join()
        self.player_thread.join()
        self.stream.close()
        self.pya.terminate()
        if self.out_file:
            self.out_file.close()

    def clear(self):
        self.b64_buffer.queue.clear()
        self.raw_buffer.queue.clear()


class SessionMode(Enum):
    SERVER_COMMIT = "server_commit"
    COMMIT = "commit"


@dataclass
class TTSConfig:
    api_key: Optional[str] = None
    model: str = "cosyvoice-v2"
    voice: str = "longhua_v2"
    format: AudioFormat = AudioFormat.MP3_22050HZ_MONO_256KBPS
    sample_rate: int = 22050
    volume: int = 50
    speech_rate: float = 1.0
    pitch_rate: float = 1.0

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ali_api_key")
        if not self.api_key:
            raise TTSException("API key not found")
        dashscope.api_key = self.api_key


@dataclass
class QwenConfig:
    api_key: Optional[str] = None
    model: str = "qwen-tts-realtime"
    voice: str = "Cherry"
    mode: SessionMode = SessionMode.SERVER_COMMIT
    sample_rate: int = 24000

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ali_api_key")
        if not self.api_key:
            raise TTSException("API key not found")
        dashscope.api_key = self.api_key


@dataclass
class SambertConfig:
    api_key: Optional[str] = None
    model: str = "sambert-zhichu-v1"
    format: str = "pcm"
    sample_rate: int = 48000
    volume: int = 50

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ali_api_key")
        if not self.api_key:
            raise TTSException("API key not found")
        dashscope.api_key = self.api_key


class CosyVoiceTTS:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._last_request_id = None
        self._first_package_delay = None

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            raise TTSException("Empty text")

        synthesizer = SpeechSynthesizer(
            model=self.config.model,
            voice=self.config.voice,
            format=self.config.format,
            volume=self.config.volume,
            speech_rate=self.config.speech_rate,
            pitch_rate=self.config.pitch_rate,
            callback=None,
        )
        audio = synthesizer.call(text)
        if not audio:
            raise TTSException("Empty audio received")

        self._last_request_id = synthesizer.get_last_request_id()
        self._first_package_delay = synthesizer.get_first_package_delay()
        self.logger.info(f"Synthesized {len(audio)} bytes")
        return audio

    def synthesize_streaming(
        self,
        text: str,
        enable_playback: bool = True,
        audio_callback: Optional[Callable[[bytes], None]] = None,
    ) -> bytes:
        if not text or not text.strip():
            raise TTSException("Empty text")

        complete_event = threading.Event()
        audio_data = []
        config = self.config

        actual_format = config.format
        if enable_playback:
            if config.format == AudioFormat.MP3_22050HZ_MONO_256KBPS:
                actual_format = AudioFormat.PCM_22050HZ_MONO_16BIT
            elif config.format == AudioFormat.MP3_24000HZ_MONO_256KBPS:
                actual_format = AudioFormat.PCM_24000HZ_MONO_16BIT
            elif config.format == AudioFormat.MP3_48000HZ_MONO_256KBPS:
                actual_format = AudioFormat.PCM_48000HZ_MONO_16BIT

        if actual_format == AudioFormat.PCM_22050HZ_MONO_16BIT:
            playback_rate = 22050
        elif actual_format == AudioFormat.PCM_24000HZ_MONO_16BIT:
            playback_rate = 24000
        elif actual_format == AudioFormat.PCM_48000HZ_MONO_16BIT:
            playback_rate = 48000
        else:
            playback_rate = 22050

        class StreamCallback(ResultCallback):
            def __init__(self):
                self.player = None
                self.stream = None
                if enable_playback:
                    self.player = pyaudio.PyAudio()
                    self.stream = self.player.open(
                        format=pyaudio.paInt16, channels=1, rate=playback_rate, output=True
                    )

            def on_open(self):
                logger.info("Stream opened")

            def on_complete(self):
                logger.info("Stream complete")
                complete_event.set()

            def on_error(self, message: str):
                logger.error(f"Stream error: {message}")
                complete_event.set()

            def on_close(self):
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                if self.player:
                    self.player.terminate()

            def on_event(self, message):
                pass

            def on_data(self, data: bytes):
                audio_data.append(data)
                if self.stream:
                    self.stream.write(data)
                if audio_callback:
                    audio_callback(data)

        cb = StreamCallback()
        synthesizer = SpeechSynthesizer(
            model=self.config.model,
            voice=self.config.voice,
            format=actual_format,
            volume=self.config.volume,
            speech_rate=self.config.speech_rate,
            pitch_rate=self.config.pitch_rate,
            callback=cb,
        )

        synthesizer.call(text)
        complete_event.wait()

        self._last_request_id = synthesizer.get_last_request_id()
        self._first_package_delay = synthesizer.get_first_package_delay()

        return b"".join(audio_data)

    def synthesize_chat(
        self,
        text_chunks: List[str],
        enable_playback: bool = True,
        audio_callback: Optional[Callable[[bytes], None]] = None,
    ) -> bytes:
        complete_event = threading.Event()
        audio_data = []
        config = self.config

        actual_format = config.format
        if enable_playback:
            if config.format == AudioFormat.MP3_22050HZ_MONO_256KBPS:
                actual_format = AudioFormat.PCM_22050HZ_MONO_16BIT
            elif config.format == AudioFormat.MP3_24000HZ_MONO_256KBPS:
                actual_format = AudioFormat.PCM_24000HZ_MONO_16BIT
            elif config.format == AudioFormat.MP3_48000HZ_MONO_256KBPS:
                actual_format = AudioFormat.PCM_48000HZ_MONO_16BIT

        if actual_format == AudioFormat.PCM_22050HZ_MONO_16BIT:
            playback_rate = 22050
        elif actual_format == AudioFormat.PCM_24000HZ_MONO_16BIT:
            playback_rate = 24000
        elif actual_format == AudioFormat.PCM_48000HZ_MONO_16BIT:
            playback_rate = 48000
        else:
            playback_rate = 22050

        class ChatCallback(ResultCallback):
            def __init__(self):
                self.player = None
                self.stream = None
                if enable_playback:
                    self.player = pyaudio.PyAudio()
                    self.stream = self.player.open(
                        format=pyaudio.paInt16, channels=1, rate=playback_rate, output=True
                    )

            def on_open(self):
                logger.info("Chat stream opened")

            def on_complete(self):
                logger.info("Chat stream complete")
                complete_event.set()

            def on_error(self, message: str):
                logger.error(f"Chat error: {message}")
                complete_event.set()

            def on_close(self):
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                if self.player:
                    self.player.terminate()

            def on_event(self, message):
                pass

            def on_data(self, data: bytes):
                audio_data.append(data)
                if self.stream:
                    self.stream.write(data)
                if audio_callback:
                    audio_callback(data)

        cb = ChatCallback()
        synthesizer = SpeechSynthesizer(
            model=self.config.model,
            voice=self.config.voice,
            format=actual_format,
            volume=self.config.volume,
            speech_rate=self.config.speech_rate,
            pitch_rate=self.config.pitch_rate,
            callback=cb,
        )

        for chunk in text_chunks:
            if chunk.strip():
                synthesizer.streaming_call(chunk)

        synthesizer.streaming_complete()
        complete_event.wait()

        self._last_request_id = synthesizer.get_last_request_id()
        self._first_package_delay = synthesizer.get_first_package_delay()

        return b"".join(audio_data)

    def save_audio(self, text: str, output_path: str) -> bool:
        try:
            audio = self.synthesize(text)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio)
            self.logger.info(f"Audio saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False

    def get_last_request_id(self) -> Optional[str]:
        return self._last_request_id

    def get_first_package_delay(self) -> Optional[int]:
        return self._first_package_delay


class QwenRealtimeTTS:
    def __init__(self, config: QwenConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.player = None
        self.qwen = None
        self.finish_event = None

    def synthesize_chat(
        self, text_chunks: List[str], enable_playback: bool = True, save_path: Optional[str] = None
    ) -> bytes:
        self.finish_event = threading.Event()

        class QwenCallback(QwenTtsRealtimeCallback):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def on_open(self):
                self.parent.logger.info("Qwen connection opened")
                if enable_playback:
                    pya = pyaudio.PyAudio()
                    self.parent.player = AudioPlayer(sample_rate=self.parent.config.sample_rate)
                    if save_path:
                        self.parent.player.set_save_file(save_path)

            def on_close(self, close_status_code, close_msg):
                self.parent.logger.info(f"Connection closed: {close_status_code}, {close_msg}")
                if self.parent.player:
                    self.parent.player.wait_complete()
                    self.parent.player.shutdown()

            def on_event(self, response: str):
                try:
                    event_type = response.get("type")
                    if event_type == "session.created":
                        self.parent.logger.info(f"Session: {response['session']['id']}")
                    elif event_type == "response.audio.delta":
                        audio_b64 = response["delta"]
                        if self.parent.player:
                            self.parent.player.add_data(audio_b64)
                    elif event_type == "response.done":
                        self.parent.logger.info("Response done")
                    elif event_type == "session.finished":
                        self.parent.logger.info("Session finished")
                        self.parent.finish_event.set()
                except Exception as e:
                    self.parent.logger.error(f"Event error: {e}")
                    self.parent.finish_event.set()

        callback = QwenCallback(self)
        self.qwen = QwenTtsRealtime(model=self.config.model, callback=callback)

        self.qwen.connect()
        self.qwen.update_session(
            voice=self.config.voice,
            response_format=QwenAudioFormat.PCM_24000HZ_MONO_16BIT,
            mode=self.config.mode.value,
        )

        for text in text_chunks:
            if text.strip():
                self.qwen.append_text(text)
                if self.config.mode == SessionMode.COMMIT:
                    self.qwen.commit()
                time.sleep(0.1)

        self.qwen.finish()
        self.finish_event.wait()
        self.qwen.close()

        self.logger.info(
            f"Session: {self.qwen.get_session_id()}, Delay: {self.qwen.get_first_audio_delay()}"
        )
        return b""


class SambertTTS:
    def __init__(self, config: SambertConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._last_request_id = None

    def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            raise TTSException("Empty text")

        result = SambertSynthesizer.call(
            model=self.config.model,
            text=text,
            sample_rate=self.config.sample_rate,
            format=self.config.format,
        )

        response = result.get_response()
        if response:
            self._last_request_id = response.get("request_id")
            self.logger.info(f"Request ID: {self._last_request_id}")

        audio = result.get_audio_data()
        if not audio:
            raise TTSException("Empty audio received")

        self.logger.info(f"Synthesized {len(audio)} bytes")
        return audio

    def synthesize_streaming(
        self,
        text: str,
        enable_playback: bool = True,
        audio_callback: Optional[Callable[[bytes], None]] = None,
    ) -> bytes:
        if not text or not text.strip():
            raise TTSException("Empty text")

        complete_event = threading.Event()
        audio_data = []
        config = self.config

        class StreamCallback(SambertCallback):
            def __init__(self):
                self.player = None
                self.stream = None
                if enable_playback:
                    self.player = pyaudio.PyAudio()
                    self.stream = self.player.open(
                        format=pyaudio.paInt16, channels=1, rate=config.sample_rate, output=True
                    )

            def on_open(self):
                logger.info("Sambert stream opened")

            def on_complete(self):
                logger.info("Sambert stream complete")
                complete_event.set()

            def on_error(self, response):
                logger.error(f"Sambert error: {response}")
                complete_event.set()

            def on_close(self):
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                if self.player:
                    self.player.terminate()

            def on_event(self, result: SpeechSynthesisResult):
                audio_frame = result.get_audio_frame()
                if audio_frame:
                    audio_data.append(audio_frame)
                    if self.stream:
                        self.stream.write(audio_frame)
                    if audio_callback:
                        audio_callback(audio_frame)

        cb = StreamCallback()
        result = SambertSynthesizer.call(
            model=self.config.model,
            text=text,
            sample_rate=self.config.sample_rate,
            format=self.config.format,
            callback=cb,
        )

        complete_event.wait()

        response = result.get_response()
        if response:
            self._last_request_id = response.get("request_id")

        return b"".join(audio_data)

    def save_audio(self, text: str, output_path: str) -> bool:
        try:
            audio = self.synthesize(text)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(audio)
            self.logger.info(f"Audio saved to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}")
            return False

    def get_last_request_id(self) -> Optional[str]:
        return self._last_request_id


class ChatTTS:
    def __init__(
        self, backend: str = "cosy", api_key: Optional[str] = None, voice: str = None, **kwargs
    ):
        self.backend = backend
        self.logger = logging.getLogger(__name__)

        if backend == "cosy":
            if voice is None:
                voice = "longhua_v2"
            config = TTSConfig(
                api_key=api_key,
                voice=voice,
                format=AudioFormat.MP3_22050HZ_MONO_256KBPS,
                sample_rate=22050,
                **kwargs,
            )
            self.engine = CosyVoiceTTS(config)
        elif backend == "qwen":
            if voice is None:
                voice = "Cherry"
            config = QwenConfig(
                api_key=api_key,
                voice=voice,
                mode=SessionMode.SERVER_COMMIT,
                sample_rate=24000,
                **kwargs,
            )
            self.engine = QwenRealtimeTTS(config)
        elif backend == "sambert":
            config = SambertConfig(api_key=api_key, **kwargs)
            self.engine = SambertTTS(config)
        else:
            raise TTSException(f"Unknown backend: {backend}")

    def speak(
        self,
        text_chunks: List[str],
        enable_playback: bool = True,
        audio_callback: Optional[Callable[[bytes], None]] = None,
        save_path: Optional[str] = None,
    ) -> bytes:
        if self.backend == "cosy":
            audio = self.engine.synthesize_chat(text_chunks, enable_playback, audio_callback)
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(audio)
            return audio
        elif self.backend == "sambert":
            text = " ".join(text_chunks)
            audio = self.engine.synthesize_streaming(text, enable_playback, audio_callback)
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(audio)
            return audio
        else:
            return self.engine.synthesize_chat(text_chunks, enable_playback, save_path)

    def speak_simple(self, text: str, enable_playback: bool = True) -> bytes:
        if self.backend in ["cosy", "sambert"]:
            if hasattr(self.engine, "synthesize"):
                return self.engine.synthesize(text)
        return self.speak([text], enable_playback)


class TTSFactory:
    @staticmethod
    def create_cosy_voice(
        api_key: Optional[str] = None,
        model: str = "cosyvoice-v2",
        voice: str = "longhua_v2",
        **kwargs,
    ) -> CosyVoiceTTS:
        config = TTSConfig(api_key=api_key, model=model, voice=voice, **kwargs)
        return CosyVoiceTTS(config)

    @staticmethod
    def create_qwen_realtime(
        api_key: Optional[str] = None,
        model: str = "qwen-tts-realtime",
        voice: str = "Cherry",
        mode: SessionMode = SessionMode.SERVER_COMMIT,
        **kwargs,
    ) -> QwenRealtimeTTS:
        config = QwenConfig(api_key=api_key, model=model, voice=voice, mode=mode, **kwargs)
        return QwenRealtimeTTS(config)

    @staticmethod
    def create_sambert(
        api_key: Optional[str] = None, model: str = "sambert-zhichu-v1", **kwargs
    ) -> SambertTTS:
        config = SambertConfig(api_key=api_key, model=model, **kwargs)
        return SambertTTS(config)

    @staticmethod
    def create_chat_tts(backend: str = "cosy", **kwargs) -> ChatTTS:
        return ChatTTS(backend=backend, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    os.makedirs("test_outputs", exist_ok=True)

    print("=" * 80)
    print("Test 1: CosyVoice Basic Synthesis")
    print("=" * 80)
    cosy = TTSFactory.create_cosy_voice()
    audio = cosy.synthesize("Hello world")
    print(f"Audio size: {len(audio)} bytes")
    cosy.save_audio("Testing save", "test_outputs/test1.mp3")

    print("\n" + "=" * 80)
    print("Test 2: CosyVoice Streaming (with playback)")
    print("=" * 80)
    print("Note: Playback auto-converts MP3 to PCM format")
    audio_stream = cosy.synthesize_streaming("Testing streaming playback", enable_playback=True)
    print(f"Streamed audio: {len(audio_stream)} bytes")
    print("Playback completed successfully")

    print("\n" + "=" * 80)
    print("Test 3: Chat TTS (CosyVoice)")
    print("=" * 80)
    chat = TTSFactory.create_chat_tts(backend="cosy", voice="longhua_v2")
    chunks = ["Hello", "this is", "a test", "of chat mode"]
    audio_chat = chat.speak(chunks, enable_playback=True, save_path="test_outputs/test_chat.mp3")
    print(f"Chat audio: {len(audio_chat)} bytes")

    print("\n" + "=" * 80)
    print("Test 4: Chat TTS Incremental")
    print("=" * 80)
    cosy_chat = TTSFactory.create_cosy_voice(format=AudioFormat.PCM_22050HZ_MONO_16BIT)
    chat_chunks = ["First chunk", "Second chunk", "Third chunk"]
    audio_inc = cosy_chat.synthesize_chat(chat_chunks, enable_playback=True)
    print(f"Incremental audio: {len(audio_inc)} bytes")

    print("\n" + "=" * 80)
    print("Test 5: Sambert TTS Basic")
    print("=" * 80)
    sambert = TTSFactory.create_sambert()
    audio_sambert = sambert.synthesize("Test Sambert synthesis")
    print(f"Sambert audio: {len(audio_sambert)} bytes")
    sambert.save_audio("Sambert file save test", "test_outputs/sambert_basic.wav")

    print("\n" + "=" * 80)
    print("Test 6: Sambert TTS Streaming")
    print("=" * 80)
    sambert_stream = TTSFactory.create_sambert(format="pcm")
    audio_sambert_stream = sambert_stream.synthesize_streaming(
        "Testing Sambert streaming", enable_playback=True
    )
    print(f"Sambert streamed audio: {len(audio_sambert_stream)} bytes")

    print("\n" + "=" * 80)
    print("Test 7: Chat TTS with Sambert Backend")
    print("=" * 80)
    chat_sambert = TTSFactory.create_chat_tts(backend="sambert")
    chunks_sambert = ["Test one", "Test two", "Test three"]
    audio_chat_sambert = chat_sambert.speak(
        chunks_sambert, enable_playback=True, save_path="test_outputs/chat_sambert.pcm"
    )
    print(f"Chat Sambert audio: {len(audio_chat_sambert)} bytes")

    print("\n" + "=" * 80)
    print("All Tests Completed")
    print("=" * 80)
