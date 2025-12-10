
# Introduction

Marvis is a cutting-edge conversational speech model designed to enable real-time voice cloning and streaming text-to-speech synthesis. Built with efficiency and accessibility in mind, Marvis addresses the growing need for high-quality, real-time voice synthesis that can run on consumer devices such as Apple Silicon and others.

Traditional voice cloning models require either the whole text input, lengthy audio samples or lack real-time streaming capabilities. Marvis bridges this gap by enabling voice cloning with just 10 seconds of audio while maintaining natural-sounding speech through intelligent text processing and streaming audio generation.

## Key Features

- **Rapid Voice Cloning**: Clone any voice using just 10 seconds of reference audio
- **Real-time Streaming**: Stream audio chunks as text is processed, enabling natural conversational flow
- **Compact Size**: Only 500MB when quantized, enabling on-device inference
- **Edge deployment**: Optimized for real-time Speech-to-Speech (STS) on mobile devices (i.e., iPad, iPhone and etc)
- **Natural Audio Flow**: Process entire text context for coherent speech synthesis without chunking artifacts
- **Multimodal Architecture**: Seamlessly handles interleaved text and audio tokens

## Supported Languages

Currently optimized for English with support for expressive speech synthesis with additional languages such as German, Portuguese, French and Mandarin coming soon.

# Quick Start

## Using MLX

```bash
pip install -U mlx-audio
python -m mlx_audio.tts.generate --model Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit --stream \
 --text "Marvis TTS is a new text-to-speech model that provides fast streaming on edge devices."
```

## Model Download

The WebSocket server uses the **Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit** model. Download it using wget:

```bash
# Create model directory
mkdir -p ~/.cache/huggingface/hub/models--Marvis-AI--marvis-tts-250m-v0.2-MLX-8bit/snapshots/main
cd ~/.cache/huggingface/hub/models--Marvis-AI--marvis-tts-250m-v0.2-MLX-8bit/snapshots/main

# Download model files manually if having issue downloading it through huggingface (for me, hangs after a while - timeout):
wget "https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit/resolve/main/config.json"
wget "https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit/resolve/main/model.safetensors"
wget "https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit/resolve/main/tokenizer.json"
wget "https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit/resolve/main/tokenizer_config.json"
wget "https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit/resolve/main/vocab.json"
```

## WebSocket Server

A WebSocket server is included for integrating Marvis TTS into other applications. The server receives text sentences via WebSocket and plays audio on system speakers.

### Web server project Structure

```
marvis-tts/
├── websocket_server_mp.py    # Main WebSocket server (multiprocessing)
├── test_continuous.py        # Test client for continuous TTS
├── voices/                   # Custom voice files directory
│   ├── conversational_a.wav  # Sample only (not used by server)
│   ├── conversational_b.wav  # Sample only (not used by server)
│   ├── your_voice.wav        # Your custom voice audio
│   └── your_voice.txt        # Transcript of your voice audio
└── README.md
```

### Installation with Conda (Recommended to select proper Python version 3.11 and libraries)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if not already installed

2. Create and activate a new conda environment:
```bash
conda create -n marvis python=3.11
conda activate marvis
```

3. Install dependencies:
```bash
pip install mlx-audio websockets sounddevice numpy
```

**Tested versions:**
- Python 3.11.14
- mlx-audio 0.2.6
- mlx 0.30.0
- websockets 15.0.1
- sounddevice 0.5.3
- numpy 2.2.6

### Starting the Server

```bash
python websocket_server_mp.py [OPTIONS]

Options:
  --host HOST           Host to bind to (default: localhost)
  --port PORT           Port to listen on (default: 8765)
  --model MODEL         Path to Marvis model (default: ~/.cache/huggingface/hub/models--Marvis-AI--marvis-tts-250m-v0.2-MLX-8bit/snapshots/main)
  --voices-dir DIR      Directory containing custom voice files (default: ./voices)
  --prebuffer SECONDS   Pre-buffer time before first playback (default: 2.0)
```

**Recommended:**
```bash
python websocket_server_mp.py --prebuffer 0
```

The server will:
1. Load the Marvis TTS model
2. Start listening on `ws://localhost:8765`
3. Accept JSON messages with text to synthesize

### WebSocket API

**Connect:** `ws://localhost:8765`

**Send messages:**
```json
{"text": "Hello, this is a test."}
{"text": "With custom voice", "voice": "your_voice_name"}
{"text": "With built-in voice", "voice": "conversational_a"}
{"text": "Faster speech", "speed": 1.2}
{"action": "list_voices"}
{"action": "set_prebuffer", "seconds": 2.0}
```

**Receive status updates:**
```json
{"status": "connected", "model": "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"}
{"status": "generating", "text": "Hello..."}
{"status": "playing", "text": "Hello...", "duration": 2.5}
{"status": "done", "text": "Hello..."}
```

### Test Clients

**Simple test:**
```bash
python test_client.py --text "Hello, this is a test."
```

**Continuous playback test (minimal gaps between sentences):**
```bash
python test_continuous.py
python test_continuous.py --text "Your long text here. It will be split into sentences."
python test_continuous.py --file document.txt
python test_continuous.py --voice your_voice_name
python test_continuous.py --voice conversational_a --speed 1.2
python test_continuous.py --prebuffer 0
```

The continuous test client sends all sentences immediately to the server's queue, allowing audio generation to happen in parallel with playback for seamless speech.

## Creating a Custom Voice

To clone your voice, you need to provide a reference audio file and its exact transcript.

### Recording Your Voice

1. Download and install [Audacity](https://www.audacityteam.org/download/)

2. **Configure audio settings BEFORE recording** (this is critical):
   - Set **Project Rate** to **24000 Hz** (bottom left corner of Audacity window)
   - Go to **Audio Setup > Recording Channels** and select **1 (Mono)**

3. Record approximately **10 seconds** of clear speech:
   - Speak naturally in a quiet environment
   - Avoid background noise
   - Use consistent volume and tone
   - Read a prepared text so you have an exact transcript

4. Export the audio:
   - File > Export > Export as WAV
   - Choose **"Signed 16-bit PCM"** encoding (this is critical)
   - Save as `your_name.wav`

5. Create a text file with the **exact transcript** of what you said:
   - Save as `your_name.txt` (same name as the WAV file)
   - The transcript must match the audio exactly word-for-word

6. Copy both files to the `voices/` directory:
   ```
   voices/
   ├── your_name.wav    # 10s, 24KHz, mono, 16-bit signed
   └── your_name.txt    # Exact transcript of the audio
   ```

**Note**: The `conversational_a.wav` and `conversational_b.wav` files in the voices directory are samples only and are NOT used by the server. These built-in voices use internal model weights.

## Reducing Hallucinations

The server uses optimized generation parameters to minimize hallucinations (random words or repeated phrases):

- **temperature**: 0.5 (default is 0.7) - Lower values produce more deterministic output
- **repetition_penalty**: 1.5 (default is 1.3) - Higher values reduce repetition

These parameters are configured in `websocket_server_mp.py` at lines 143-144.

**Important**: When using custom voices, you may need to adjust these parameters if you experience hallucinations. Try:
- Lowering temperature further (0.3-0.5)
- Increasing repetition_penalty (1.5-2.0)
- Ensuring your reference audio is exactly 10 seconds
- Making sure the transcript in your `.txt` file exactly matches the audio

## Using transformers

**Without Voice Cloning**
```python
import torch
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from tokenizers.processors import TemplateProcessing
import soundfile as sf

model_id = "Marvis-AI/marvis-tts-250m-v0.1"
device = "cuda"if torch.cuda.is_available() else "cpu"

# load the model and the processor
processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

# prepare the inputs
text = "[0]Marvis TTS is a new text-to-speech model that provides fast streaming on edge devices." # `[0]` for speaker id 0
inputs = processor(text, add_special_tokens=True, return_tensors="pt").to(device).pop("token_type_ids")
# infer the model
audio = model.generate(**inputs, output_audio=True)
sf.write("example_without_context.wav", audio[0].cpu(), samplerate=24_000, subtype="PCM_16")

```


# Model Description

Marvis is built on the [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) (Conversational Speech Model) architecture, a multimodal transformer that operates directly on Residual Vector Quantization (RVQ) tokens and uses [Kyutai's mimi codec](https://huggingface.co/kyutai/mimi). The architecture enables end-to-end training while maintaining low-latency generation and employs a dual-transformer approach:

- **Multimodal Backbone (250M parameters)**: Processes interleaved text and audio sequences to model the zeroth codebook level, providing semantic understanding and context.

- **Audio Decoder (60M parameters)**: A smaller, specialized transformer that models the remaining 31 codebook levels to reconstruct high-quality speech from the backbone's representations.

Unlike models that require text chunking based on regex patterns, Marvis processes entire text sequences contextually, resulting in more natural speech flow and intonation.

**Key Architectural Innovation**: Unlike models that require text chunking based on regex patterns, Marvis processes entire text sequences contextually, resulting in more natural speech flow and intonation.

# Training Details

**Pretraining**:
- Dataset: Emilia-YODAS
- Training Steps: 2M steps
- Hardware: 1x NVIDIA GH200 96GB
- Precision: bfloat16
- Learning Rate: 3e-4
- Batch Size: 64

**Post-training**:
- Dataset: Expressive Speech
- Training Steps: 200K steps
- Expressiveness Setting: 0.5
- Hardware: 1x NVIDIA GH200 96GB
- Precision: bfloat16
- Learning Rate: 1e-4
- Batch Size: 64

**Total Training Cost**: ~$2,000
- Pretraining and fine-tuning: $246.69 (1x GH200)
- Post-training data generation: $167.94 (RTX6000 Ada)
- Additional experimentation: ~$1,500 across various GPU configurations
- Platforms: Prime-Intellect and Jarvis-Labs

## Use Cases

- **Real-time Voice Assistants**: Deploy natural-sounding voice interfaces with custom voices
- **Content Creation**: Generate voiceovers and narration with personalized voices
- **Accessibility Tools**: Create personalized speech synthesis for communication aids
- **Interactive Applications**: Build conversational AI with consistent voice identity
- **Podcast & Media**: Generate natural-sounding speech for automated content

### Local & Cloud Deployment

**Local Deployment:**
- Minimum Requirements: 1GB RAM, GPU recommended for real-time inference
- Quantized Model: 500MB download
- Platforms: iOS, Android, Windows, macOS, Linux

**Cloud Deployment:**
- API-ready architecture
- Scalable inference pipeline
- Low-latency streaming support

### Technical Limitations

- Language Support: Currently optimized primarily for English. Performance on other languages may be suboptimal
- Audio Quality Dependency: Voice cloning quality is dependent on the clarity and quality of the 10-second reference audio
- Background Noise: Performance degrades with noisy reference audio or inference environments
- Hallucinations: The model might hallucinate words specially for new words or short sentences.

### Legal and Ethical Considerations:

- Users are responsible for complying with local laws regarding voice synthesis and impersonation
- Consider intellectual property rights when cloning voices of public figures
- Respect privacy laws and regulations in your jurisdiction
- Obtain appropriate consent and permissions before deployment

## License & Agreement

* Apache 2.0

## Citation

If you use Marvis in your research or applications, please cite:

```bibtex
@misc{marvis-tts-2025,
  title={Marvis-TTS: Efficient Real-time Voice Cloning with Streaming Speech Synthesis},
  author={Prince Canuma and Lucas Newman},
  year={2025}
}
```

## Acknowledgments

Special thanks to Sesame and Kyutai for their groundbreaking open-source contributions that inspired our work, and to the broader open-source community for their unwavering support and collaboration.

---

**Version**: 0.2

**Release Date**: 26/08/2025

**Creators**: Prince Canuma & Lucas Newman
