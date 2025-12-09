#!/usr/bin/env python3
"""
Test client for WebSocket TTS Server.

Usage:
    python test_client.py
    python test_client.py --text "Custom text to speak"
    python test_client.py --file document.txt
"""

import argparse
import asyncio
import json

import websockets


async def send_sentences(uri: str, sentences: list[str]):
    """Connect to WebSocket server and send sentences."""
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as websocket:
        # Wait for connection message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Connected: {data.get('message', 'OK')}")
        print(f"Model: {data.get('model', 'unknown')}")
        print("-" * 50)

        for sentence in sentences:
            if not sentence.strip():
                continue

            print(f"\nSending: {sentence[:60]}{'...' if len(sentence) > 60 else ''}")

            # Send the sentence
            await websocket.send(json.dumps({"text": sentence}))

            # Wait for responses
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                status = data.get("status")

                if status == "generating":
                    print("  Status: Generating...")
                elif status == "playing":
                    duration = data.get("duration", 0)
                    print(f"  Status: Playing ({duration}s)")
                elif status == "done":
                    print("  Status: Done")
                    break
                elif status == "error":
                    print(f"  Error: {data.get('error')}")
                    break

        print("\n" + "-" * 50)
        print("All sentences sent!")


def main():
    parser = argparse.ArgumentParser(description="Test client for WebSocket TTS Server")
    parser.add_argument("--uri", default="ws://localhost:8765", help="WebSocket URI")
    parser.add_argument("--text", help="Text to speak")
    parser.add_argument("--file", help="File to read sentences from")

    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            text = f.read()
        # Split into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
    elif args.text:
        sentences = [args.text]
    else:
        # Default test sentences
        sentences = [
            "Hello, this is a test of the WebSocket TTS server.",
            "The quick brown fox jumps over the lazy dog.",
            "Marvis TTS provides real-time speech synthesis on Apple Silicon.",
        ]

    asyncio.run(send_sentences(args.uri, sentences))


if __name__ == "__main__":
    main()
