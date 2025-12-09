#!/usr/bin/env python3
"""
Continuous TTS Test Client - Sends sentences rapidly to minimize gaps.

This client splits text into sentences and sends them immediately to the
WebSocket server, allowing the server to queue audio generation while
previous sentences are still playing.

Usage:
    python test_continuous.py
    python test_continuous.py --text "Your long text here..."
    python test_continuous.py --file document.txt
"""

import argparse
import asyncio
import json
import re
import time

import websockets


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty sentences and strip whitespace
    return [s.strip() for s in sentences if s.strip()]


async def continuous_tts(uri: str, text: str):
    """
    Send all sentences immediately, then wait for all to complete.
    This keeps the server's audio queue filled for seamless playback.
    """
    sentences = split_into_sentences(text)

    if not sentences:
        print("No sentences to process.")
        return

    print(f"Text split into {len(sentences)} sentences")
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        # Wait for connection message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Connected: {data.get('message', 'OK')}")
        print(f"Model: {data.get('model', 'unknown')}")
        print("=" * 60)

        # Track pending sentences
        pending = set()
        completed = 0
        total_audio_duration = 0
        start_time = time.time()

        # Send ALL sentences immediately (don't wait for responses)
        print("\nSending all sentences to queue...")
        for i, sentence in enumerate(sentences):
            await websocket.send(json.dumps({"text": sentence}))
            pending.add(sentence[:50])  # Use first 50 chars as key
            print(f"  [{i+1}/{len(sentences)}] Queued: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")

        send_time = time.time() - start_time
        print(f"\nAll {len(sentences)} sentences queued in {send_time:.2f}s")
        print("=" * 60)
        print("\nPlaying audio (listening for status updates)...\n")

        # Now listen for all responses
        while completed < len(sentences):
            response = await websocket.recv()
            data = json.loads(response)
            status = data.get("status")
            text_snippet = data.get("text", "")[:50]

            if status == "generating":
                print(f"  Generating: {text_snippet}...")
            elif status == "playing":
                duration = data.get("duration", 0)
                total_audio_duration += duration
                print(f"  Playing ({duration:.1f}s): {text_snippet}...")
            elif status == "done":
                completed += 1
                print(f"  Done [{completed}/{len(sentences)}]: {text_snippet}...")
            elif status == "error":
                completed += 1
                print(f"  ERROR: {data.get('error')}")

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Sentences processed: {len(sentences)}")
        print(f"Total audio duration: {total_audio_duration:.2f}s")
        print(f"Total wall time: {total_time:.2f}s")
        print(f"Overhead: {total_time - total_audio_duration:.2f}s")


# Sample long text for testing
SAMPLE_TEXT = """
Marvis is a cutting-edge conversational speech model designed to enable real-time voice cloning and streaming text-to-speech synthesis. Built with efficiency and accessibility in mind, Marvis addresses the growing need for high-quality, real-time voice synthesis that can run on consumer devices such as Apple Silicon and others.

Traditional voice cloning models require either the whole text input, lengthy audio samples or lack real-time streaming capabilities. Marvis bridges this gap by enabling voice cloning with just 10 seconds of audio while maintaining natural-sounding speech through intelligent text processing and streaming audio generation.

The model features rapid voice cloning using just 10 seconds of reference audio. It supports real-time streaming, allowing audio chunks to be generated as text is processed. This enables natural conversational flow in applications like voice assistants and interactive systems.

With a compact size of only 500 megabytes when quantized, Marvis enables on-device inference. It's optimized for edge deployment, providing real-time speech-to-speech capabilities on mobile devices including iPad and iPhone.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Continuous TTS test - sends sentences rapidly to minimize gaps"
    )
    parser.add_argument("--uri", default="ws://localhost:8765", help="WebSocket URI")
    parser.add_argument("--text", help="Text to speak (will be split into sentences)")
    parser.add_argument("--file", help="File to read text from")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = SAMPLE_TEXT

    asyncio.run(continuous_tts(args.uri, text))


if __name__ == "__main__":
    main()
