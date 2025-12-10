# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Install dependencies: `pip install mlx-audio websockets sounddevice numpy`
- Start WebSocket server: `python websocket_server_mp.py --prebuffer 0`
- Run test client: `python test_continuous.py --voice your_voice_name`

## Code Style Guidelines
- Import order: standard lib, third-party, project-specific
- Use Python 3.10+ features and type hints
- Follow PEP 8 naming conventions
- Audio processing uses 24000Hz sample rate

## Key Parameters for Voice Cloning
- temperature: 0.5 (lower = more deterministic, reduces hallucinations)
- repetition_penalty: 1.5 (higher = less repetition, reduces hallucinations)
- ref_audio: Must be ~10s, 24000Hz, mono, 16-bit signed WAV
- ref_text: Exact transcript of the reference audio
