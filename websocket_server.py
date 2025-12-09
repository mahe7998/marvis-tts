#!/usr/bin/env python3
"""
WebSocket TTS Server for Marvis.

Generates audio in a separate thread while playing, minimizing gaps between sentences.
Receives text sentences via WebSocket and plays audio on system speakers.

Usage:
    python websocket_server.py [--port 8765]

WebSocket API:
    Connect: ws://localhost:8765
    Send: {"text": "Your sentence here"}
    Receive: {"status": "generating/playing/done", "text": "..."}
"""

import argparse
import asyncio
import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

import websockets

# Configure sounddevice for higher latency to avoid buffer underruns
# This helps when CPU is busy with TTS generation
sd.default.latency = 1.0  # 1 second latency buffer
sd.default.blocksize = 8192  # Larger block size for smoother playback

from mlx_audio.utils import load_model as load_tts_model


@dataclass
class TTSRequest:
    text: str
    voice: Optional[str] = None
    speed: float = 1.0
    websocket: Optional[any] = None


@dataclass
class AudioItem:
    audio: np.ndarray
    sample_rate: int
    text: str
    websocket: Optional[any] = None
    duration: float = 0


class MarvisTTSServer:
    def __init__(self, model_name: str = "Marvis-AI/marvis-tts-250m-v0.1-MLX-8bit"):
        self.model_name = model_name
        self.model = None

        # Two separate queues: one for generation requests, one for audio playback
        self.generation_queue = queue.Queue()
        self.playback_queue = queue.Queue()

        self.is_running = True
        self.generation_thread = None
        self.playback_thread = None

        # Event loop reference for async callbacks
        self.loop = None

    def load_model(self):
        """Load the Marvis TTS model."""
        print(f"Loading model: {self.model_name}")
        self.model = load_tts_model(self.model_name)
        print("Model loaded!")

    def start_threads(self):
        """Start generation and playback threads."""
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.generation_thread.start()
        self.playback_thread.start()

    def _generation_loop(self):
        """Background thread that generates audio from text requests."""
        while self.is_running:
            try:
                request = self.generation_queue.get(timeout=1)
                if request is None:
                    break

                start_time = time.time()

                # Generate audio
                for result in self.model.generate(
                    request.text,
                    voice=request.voice,
                    speed=request.speed,
                ):
                    audio = result.audio
                    sample_rate = result.sample_rate

                gen_time = time.time() - start_time
                duration = len(audio) / sample_rate

                print(f"Generated: '{request.text[:40]}...' ({duration:.1f}s audio in {gen_time:.2f}s)")

                # Send "playing" status
                if request.websocket and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self._send_status(request.websocket, "playing", request.text, duration),
                        self.loop
                    )

                # Queue for playback
                self.playback_queue.put(AudioItem(
                    audio=audio,
                    sample_rate=sample_rate,
                    text=request.text,
                    websocket=request.websocket,
                    duration=duration
                ))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Generation error: {e}")

    def _playback_loop(self):
        """Background thread that plays audio from the queue."""
        first_play = True
        while self.is_running:
            try:
                item = self.playback_queue.get(timeout=1)
                if item is None:
                    break

                # On first playback, wait a moment to let generation get ahead
                # This helps prevent buffer underruns from CPU contention
                if first_play:
                    time.sleep(0.5)  # 500ms head start for buffer to fill
                    first_play = False

                # Play audio with blocking wait
                sd.play(item.audio, item.sample_rate)
                sd.wait()

                # Send "done" status
                if item.websocket and self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self._send_status(item.websocket, "done", item.text),
                        self.loop
                    )

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Playback error: {e}")

    async def _send_status(self, websocket, status: str, text: str, duration: float = 0):
        """Send status message to websocket client."""
        try:
            msg = {"status": status, "text": text}
            if duration > 0:
                msg["duration"] = round(duration, 2)
            await websocket.send(json.dumps(msg))
        except Exception as e:
            print(f"WebSocket send error: {e}")

    async def handle_websocket(self, websocket):
        """Handle incoming WebSocket connections."""
        client_addr = websocket.remote_address
        print(f"Client connected: {client_addr}")

        try:
            await websocket.send(json.dumps({
                "status": "connected",
                "model": self.model_name,
                "message": "Send JSON with 'text' field to generate speech"
            }))

            async for message in websocket:
                try:
                    data = json.loads(message)

                    if "text" not in data:
                        await websocket.send(json.dumps({
                            "status": "error",
                            "error": "Missing 'text' field"
                        }))
                        continue

                    text = data["text"].strip()
                    if not text:
                        continue

                    voice = data.get("voice")
                    speed = data.get("speed", 1.0)

                    # Acknowledge receipt immediately
                    await websocket.send(json.dumps({
                        "status": "generating",
                        "text": text
                    }))

                    # Queue for generation (non-blocking)
                    request = TTSRequest(
                        text=text,
                        voice=voice,
                        speed=speed,
                        websocket=websocket
                    )
                    self.generation_queue.put(request)

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "status": "error",
                        "error": "Invalid JSON"
                    }))

        except websockets.exceptions.ConnectionClosed:
            print(f"Client disconnected: {client_addr}")
        except Exception as e:
            print(f"WebSocket error: {e}")

    def stop(self):
        """Stop the server."""
        self.is_running = False
        self.generation_queue.put(None)
        self.playback_queue.put(None)
        if self.generation_thread:
            self.generation_thread.join(timeout=5)
        if self.playback_thread:
            self.playback_thread.join(timeout=5)


async def main(host: str, port: int, model: str):
    """Main server entry point."""
    server = MarvisTTSServer(model_name=model)

    # Store event loop reference for thread callbacks
    server.loop = asyncio.get_event_loop()

    # Pre-load the model
    server.load_model()

    # Start generation and playback threads
    server.start_threads()

    print(f"\nWebSocket TTS Server starting on ws://{host}:{port}")
    print("=" * 60)
    print("Connect and send:")
    print('  {"text": "Hello, this is a test."}')
    print("=" * 60)

    try:
        async with websockets.serve(server.handle_websocket, host, port):
            await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket TTS Server for Marvis")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--model",
        default="Marvis-AI/marvis-tts-250m-v0.1-MLX-8bit",
        help="Marvis model to use"
    )

    args = parser.parse_args()
    asyncio.run(main(args.host, args.port, args.model))
