#!/usr/bin/env python3
"""
WebSocket TTS Server for Marvis - Multiprocessing Version.

Uses separate PROCESSES for generation and playback to avoid GIL contention.
This allows true parallel execution of TTS generation and audio playback.

Usage:
    python websocket_server_mp.py [--port 8765]
    python websocket_server_mp.py --voices-dir ./voices

WebSocket API:
    Connect: ws://localhost:8765
    Send: {"text": "Your sentence here"}
    Send: {"text": "Hello", "voice": "my_custom_voice"}  # Custom voice from voices/ dir
    Send: {"action": "list_voices"}  # List available voices
    Send: {"action": "set_prebuffer", "seconds": 2.0}  # Set pre-buffer time (default: 0s)
    Receive: {"status": "generating/playing/done", "text": "..."}
"""

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("main")

# Configure sounddevice for higher latency to avoid buffer underruns
sd.default.latency = 1.0
sd.default.blocksize = 8192


@dataclass
class TTSRequest:
    text: str
    voice: Optional[str] = None
    ref_audio_path: Optional[str] = None  # Path to custom voice audio file
    ref_text: Optional[str] = None
    speed: float = 1.0
    request_id: int = 0


def generation_process(request_queue: mp.Queue, audio_queue: mp.Queue, model_path: str, voices_dir: str):
    """
    Separate PROCESS for TTS generation.
    Runs independently with its own Python interpreter - no GIL contention with playback.
    """
    proc_logger = logging.getLogger("gen")
    proc_logger.info(f"Generation process started (PID: {os.getpid()})")

    # Import heavy dependencies only in this process
    from mlx_audio.utils import load_model as load_tts_model
    from mlx_audio.tts.generate import load_audio

    # Load model
    model_path_expanded = os.path.expanduser(model_path)
    proc_logger.info(f"Loading model: {model_path_expanded}")
    model = load_tts_model(model_path_expanded)
    proc_logger.info("Model loaded!")

    # Voice cache for this process
    voice_cache = {}
    voices_path = Path(voices_dir)

    def get_cached_voice(voice_name: str):
        if voice_name in voice_cache:
            return voice_cache[voice_name]

        voice_path = voices_path / f"{voice_name}.wav"
        text_path = voices_path / f"{voice_name}.txt"

        if not voice_path.exists() or not text_path.exists():
            return None

        proc_logger.info(f"Caching voice: {voice_name}")
        audio_data = load_audio(str(voice_path), sample_rate=24000)
        ref_text = text_path.read_text().strip()

        voice_cache[voice_name] = {
            'audio_data': audio_data,
            'ref_text': ref_text
        }
        return voice_cache[voice_name]

    while True:
        try:
            request = request_queue.get(timeout=1)
            if request is None:
                proc_logger.info("Generation process received stop signal")
                break

            proc_logger.info(f"[{request.request_id}] Generating: '{request.text[:50]}...'")
            start_time = time.time()

            # Build generation kwargs
            gen_kwargs = {
                "text": request.text,
                "speed": request.speed,
            }

            # Handle voice
            if request.ref_audio_path:
                cached = get_cached_voice(Path(request.ref_audio_path).stem)
                if cached:
                    gen_kwargs["ref_audio"] = cached['audio_data']
                    gen_kwargs["ref_text"] = cached['ref_text']
                else:
                    # Load directly
                    gen_kwargs["ref_audio"] = load_audio(request.ref_audio_path, sample_rate=24000)
                    if request.ref_text:
                        gen_kwargs["ref_text"] = request.ref_text
            elif request.voice:
                # Check if it's a custom voice
                cached = get_cached_voice(request.voice)
                if cached:
                    gen_kwargs["ref_audio"] = cached['audio_data']
                    gen_kwargs["ref_text"] = cached['ref_text']
                else:
                    gen_kwargs["voice"] = request.voice

            # Generate audio with lower temperature and higher repetition penalty
            # to reduce hallucinations
            gen_kwargs["temperature"] = 0.5  # Default is 0.7
            gen_kwargs["repetition_penalty"] = 1.5  # Default is 1.3

            for result in model.generate(**gen_kwargs):
                audio = result.audio
                sample_rate = result.sample_rate

            # Convert to numpy for pickling across processes
            audio_np = np.array(audio)

            gen_time = time.time() - start_time
            duration = len(audio_np) / sample_rate

            proc_logger.info(f"[{request.request_id}] Generated: {duration:.1f}s audio in {gen_time:.2f}s")

            # Put audio in queue for playback process
            audio_queue.put({
                'audio': audio_np,
                'sample_rate': sample_rate,
                'text': request.text,
                'duration': duration,
                'request_id': request.request_id
            })

        except queue.Empty:
            continue
        except Exception as e:
            proc_logger.error(f"Generation error: {e}", exc_info=True)


def playback_process(audio_queue: mp.Queue, status_queue: mp.Queue, prebuffer_value: mp.Value):
    """
    Separate PROCESS for audio playback.
    Runs independently - no GIL contention with generation.
    """
    proc_logger = logging.getLogger("play")
    proc_logger.info(f"Playback process started (PID: {os.getpid()})")

    first_play = True

    while True:
        try:
            item = audio_queue.get(timeout=1)
            if item is None:
                proc_logger.info("Playback process received stop signal")
                break

            # Pre-buffer delay on first playback
            prebuffer_seconds = prebuffer_value.value
            if first_play and prebuffer_seconds > 0:
                proc_logger.info(f"First playback - waiting {prebuffer_seconds}s for buffer")
                time.sleep(prebuffer_seconds)
                proc_logger.info(f"Prebuffer wait complete, starting playback")
                first_play = False

            # Send "playing" status
            status_queue.put({
                'status': 'playing',
                'text': item['text'],
                'duration': item['duration'],
                'request_id': item['request_id']
            })

            # Play audio
            proc_logger.info(f"[{item['request_id']}] Playing: {item['duration']:.1f}s")
            sd.play(item['audio'], item['sample_rate'])
            sd.wait()

            # Send "done" status
            status_queue.put({
                'status': 'done',
                'text': item['text'],
                'request_id': item['request_id']
            })

            proc_logger.info(f"[{item['request_id']}] Done")

        except queue.Empty:
            continue
        except Exception as e:
            proc_logger.error(f"Playback error: {e}", exc_info=True)


class MarvisTTSServerMP:
    """Multiprocessing-based TTS server."""

    BUILTIN_VOICES = ["conversational_a", "conversational_b"]

    def __init__(self, model_name: str, voices_dir: str = None, prebuffer: float = 2.0):
        self.model_name = model_name
        self.voices_dir = Path(voices_dir) if voices_dir else Path(__file__).parent / "voices"

        # Multiprocessing queues
        self.request_queue = mp.Queue()
        self.audio_queue = mp.Queue()
        self.status_queue = mp.Queue()

        # Shared value for prebuffer (can be updated by client)
        self.prebuffer_value = mp.Value('d', prebuffer)  # 'd' = double (float)

        # Processes
        self.gen_process = None
        self.play_process = None

        # Request ID counter
        self.request_id = 0

        # WebSocket tracking
        self.pending_requests = {}  # request_id -> websocket

        # Event loop reference
        self.loop = None
        self.is_running = True

    def get_available_voices(self) -> dict:
        voices = {"builtin": self.BUILTIN_VOICES.copy(), "custom": []}
        if self.voices_dir.exists():
            for wav_file in self.voices_dir.glob("*.wav"):
                voice_name = wav_file.stem
                if voice_name not in self.BUILTIN_VOICES:
                    voices["custom"].append(voice_name)
        return voices

    def get_voice_path(self, voice_name: str) -> Optional[str]:
        if voice_name in self.BUILTIN_VOICES:
            return None
        voice_path = self.voices_dir / f"{voice_name}.wav"
        if voice_path.exists():
            return str(voice_path)
        return None

    def get_voice_text(self, voice_name: str) -> Optional[str]:
        text_path = self.voices_dir / f"{voice_name}.txt"
        if text_path.exists():
            return text_path.read_text().strip()
        return None

    def start_processes(self):
        """Start generation and playback processes."""
        logger.info("Starting generation process...")
        self.gen_process = mp.Process(
            target=generation_process,
            args=(self.request_queue, self.audio_queue, self.model_name, str(self.voices_dir)),
            daemon=True
        )
        self.gen_process.start()

        logger.info("Starting playback process...")
        self.play_process = mp.Process(
            target=playback_process,
            args=(self.audio_queue, self.status_queue, self.prebuffer_value),
            daemon=True
        )
        self.play_process.start()

        # Start status monitor thread (lightweight, just forwards messages)
        self.status_thread = threading.Thread(target=self._status_monitor, daemon=True)
        self.status_thread.start()

    def _status_monitor(self):
        """Monitor status queue and forward to websockets."""
        logger.info("Status monitor started")
        while self.is_running:
            try:
                status = self.status_queue.get(timeout=1)
                request_id = status.get('request_id')
                websocket = self.pending_requests.get(request_id)

                if websocket and self.loop:
                    msg = {"status": status['status'], "text": status['text']}
                    if 'duration' in status:
                        msg['duration'] = round(status['duration'], 2)

                    asyncio.run_coroutine_threadsafe(
                        self._send_json(websocket, msg),
                        self.loop
                    )

                    if status['status'] == 'done':
                        self.pending_requests.pop(request_id, None)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Status monitor error: {e}")

    async def _send_json(self, websocket, msg):
        try:
            await websocket.send(json.dumps(msg))
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")

    async def handle_websocket(self, websocket):
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        try:
            await websocket.send(json.dumps({
                "status": "connected",
                "model": self.model_name,
                "message": "Send JSON with 'text' field to generate speech"
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)

                    # Handle list_voices
                    if data.get("action") == "list_voices":
                        voices = self.get_available_voices()
                        await websocket.send(json.dumps({"status": "voices", "voices": voices}))
                        continue

                    # Handle set_prebuffer
                    if data.get("action") == "set_prebuffer":
                        seconds = data.get("seconds", 2.0)
                        new_prebuffer = max(0, min(10, float(seconds)))
                        self.prebuffer_value.value = new_prebuffer
                        await websocket.send(json.dumps({
                            "status": "prebuffer_set",
                            "seconds": new_prebuffer
                        }))
                        continue

                    if "text" not in data:
                        await websocket.send(json.dumps({"status": "error", "error": "Missing 'text' field"}))
                        continue

                    text = data["text"].strip()
                    if not text:
                        continue

                    voice = data.get("voice")
                    speed = data.get("speed", 1.0)
                    ref_audio_path = None
                    ref_text = None

                    # Check voice type
                    if voice:
                        voice_path = self.get_voice_path(voice)
                        if voice_path:
                            ref_text = self.get_voice_text(voice)
                            if not ref_text:
                                await websocket.send(json.dumps({
                                    "status": "error",
                                    "error": f"Missing transcript file: {voice}.txt"
                                }))
                                continue
                            ref_audio_path = voice_path
                            voice = None  # Use ref_audio instead
                        elif voice not in self.BUILTIN_VOICES:
                            await websocket.send(json.dumps({
                                "status": "error",
                                "error": f"Unknown voice: {voice}"
                            }))
                            continue

                    # Assign request ID
                    self.request_id += 1
                    req_id = self.request_id

                    # Track websocket for this request
                    self.pending_requests[req_id] = websocket

                    # Acknowledge
                    await websocket.send(json.dumps({"status": "generating", "text": text}))

                    # Queue request
                    request = TTSRequest(
                        text=text,
                        voice=voice,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                        speed=speed,
                        request_id=req_id
                    )
                    self.request_queue.put(request)
                    logger.info(f"[{req_id}] Queued: '{text[:50]}...'")

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({"status": "error", "error": "Invalid JSON"}))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)

    def stop(self):
        self.is_running = False
        self.request_queue.put(None)
        self.audio_queue.put(None)
        if self.gen_process:
            self.gen_process.join(timeout=5)
        if self.play_process:
            self.play_process.join(timeout=5)


async def main(host: str, port: int, model: str, voices_dir: str = None, prebuffer: float = 2.0):
    server = MarvisTTSServerMP(model_name=model, voices_dir=voices_dir, prebuffer=prebuffer)
    server.loop = asyncio.get_event_loop()

    # Start processes
    server.start_processes()

    # Wait for generation process to load model
    logger.info("Waiting for model to load...")
    time.sleep(3)  # Give time for model to load

    voices = server.get_available_voices()

    print(f"\nWebSocket TTS Server (MP) starting on ws://{host}:{port}")
    print("=" * 60)
    print(f"Pre-buffer: {server.prebuffer_value.value}s (default, can be changed via API)")
    print("Available voices:")
    print(f"  Built-in: {', '.join(voices['builtin'])}")
    if voices['custom']:
        print(f"  Custom:   {', '.join(voices['custom'])}")
    else:
        print(f"  Custom:   (none - add .wav files to {server.voices_dir})")
    print()
    print("Connect and send:")
    print('  {"text": "Hello, this is a test."}')
    print('  {"text": "Hello", "voice": "conversational_b"}')
    print('  {"action": "list_voices"}')
    print("=" * 60)

    try:
        async with websockets.serve(server.handle_websocket, host, port):
            await asyncio.Future()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="WebSocket TTS Server for Marvis (Multiprocessing)")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--model",
        default="~/.cache/huggingface/hub/models--Marvis-AI--marvis-tts-250m-v0.2-MLX-8bit/snapshots/main",
        help="Marvis model to use"
    )
    parser.add_argument("--voices-dir", default=None, help="Directory containing custom voice files")
    parser.add_argument("--prebuffer", type=float, default=2.0, help="Pre-buffer time in seconds (default: 2.0)")

    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, args.model, args.voices_dir, args.prebuffer))
