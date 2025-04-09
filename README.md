# ComfyUI-MegaTTS

A ComfyUI custom node based on ByteDance [MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3), enabling high-quality text-to-speech synthesis with voice cloning capabilities for both Chinese and English.
![comfyui-megaTTS](https://github.com/user-attachments/assets/c395e3f8-099f-4471-bfa3-2d94d682a7cf)


## Features

- **High-Quality Voice Synthesis**: Generate natural-sounding speech from text input
- **Voice Cloning**: Clone any voice with just a short sample (requires both WAV and NPY files)
- **Bilingual Support**: Works with both Chinese and English text, with code-switching capabilities
- **Advanced Parameter Control**: Fine-tune generation quality, pronunciation accuracy, and voice similarity
- **Memory Management**: Built-in functionality to optimize GPU resource usage
- **Automatic Model Download**: Models are downloaded automatically when required

## Installation

### Prerequisites

- ComfyUI installed and working
- Python 3.10+ recommended
- CUDA-compatible GPU with at least 4GB VRAM (8GB+ recommended for higher quality)

### Steps

1. Clone this repository to ComfyUI's `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/gordon123/ComfyUI-MegaTTS3-a.git
   ```

2. Install required dependencies:
   ```bash
   cd ComfyUI-MegaTTS3-a
   pip install -r requirements.txt
   ```

3. The node will automatically download required models on first use, in this folder "comfyui/models/TTS/MegaTTS3/" or you can manually below:

## Manual Models  Download 

This extension uses modified versions of ByteDance's MegaTTS3 models. While the models are automatically downloaded during first use, you can manually download them from Hugging Face:

### Model Structure

The models are organized in the following structure:
```
comfyui/models/TTS/MegaTTS3/
├── aligner_lm/            # Speech-text alignment model
├── diffusion_transformer/ # Main TTS model
├── duration_lm/           # Duration prediction model
├── g2p/                   # Grapheme-to-phoneme model
└── wavvae/                # WaveVAE vocoder
```

### Manual Download Options

1. **Direct Download from Hugging Face**:
   - Visit the [ByteDance/MegaTTS3 repository](https://huggingface.co/ByteDance/MegaTTS3/tree/main)
   - Download each subfolder from the repository:
     - [aligner_lm](https://huggingface.co/ByteDance/MegaTTS3/tree/main/aligner_lm)
     - [diffusion_transformer](https://huggingface.co/ByteDance/MegaTTS3/tree/main/diffusion_transformer)
     - [duration_lm](https://huggingface.co/ByteDance/MegaTTS3/tree/main/duration_lm)
     - [g2p](https://huggingface.co/ByteDance/MegaTTS3/tree/main/g2p)
     - [wavvae](https://huggingface.co/ByteDance/MegaTTS3/tree/main/wavvae)
   - Place the downloaded files in the corresponding directories under `comfyui/models/TTS/MegaTTS3/`

2. **Using Hugging Face CLI**:
   ```bash
   # Install huggingface_hub if you don't have it
   pip install huggingface_hub
   
   # Download all models
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ByteDance/MegaTTS3', local_dir='comfyui/models/TTS/MegaTTS3/')"
   ```

## Voice Folder and Voice Maker
![Voice_Maker](https://github.com/user-attachments/assets/b3713f9a-2f70-4bf0-a6f4-7e3110bba987)

### Voice Folder Structure

The extension requires a `Voices` folder to store reference voice samples and their extracted features:

```
Voices/
├── sample1.wav     # Reference audio file
├── sample1.npy     # Extracted features from the audio file
├── sample2.wav     # Another reference audio
└── sample2.npy     # Corresponding features
```

### Voice Maker Node

This extension includes a **Voice Maker** custom node that helps you prepare voice samples:

- **Voice Maker Node Features**:
  - Convert any audio file to the required 24kHz WAV format
  - Extract NPY feature files from WAV samples
  - Process and optimize voice samples for better quality
  - Save processed files to the Voices folder automatically

**How to use the Voice Maker**:
1. Add the "Voice Maker" node from the 🧪AILab/🔊Audio category
2. Connect an audio input or select a file from your computer
3. Configure processing options (normalization, trimming, etc.)
4. Run the node to generate a ready-to-use voice sample with its NPY file

### About WAV and NPY Files

- **WAV files**: These are the actual voice samples you want to clone (24kHz recommended)
- **NPY files**: These contain extracted features necessary for voice cloning

### Important Note on NPY Files

As noted in the [original MegaTTS3 repository](https://github.com/bytedance/MegaTTS3):

> For security issues, we do not upload the parameters of WaveVAE encoder to the above links. You can only use the pre-extracted latents (.npy files) for inference. If you want to synthesize speech for speaker A, you need "A.wav" and "A.npy" in the same directory.

### Getting Voice Samples and NPY Files

1. **Download pre-extracted samples**:
   - Sample voice WAV and NPY files can be found in this Google Drive folder: [Voice Samples and NPY Files](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr?usp=sharing)
   - This folder contains pre-extracted NPY files and their corresponding WAV samples organized in subfolders

2. **Submit your own voice samples**:
   - If you want to use your own voice, you can submit samples to this Google Drive folder: [Voice Submission Queue](https://drive.google.com/drive/folders/1gCWL1y_2xu9nIFhUX_OW5MbcFuB7J5Cl?usp=sharing)
   - Your samples should be clear audio with minimal background noise and within 24 seconds
   - After verification for safety, the ByteDance team will extract and provide NPY files for your samples

3. **Generate NPY files with Voice Maker**:
   - Use the Voice Maker node to automatically process your audio and generate NPY files
   - While this method is convenient, the quality may not match officially extracted NPY files
   - Best for quick testing and experimentation with your own voice samples

### Voice Format Requirements

For best results:
- **Sample rate**: 24kHz (will be automatically converted if different)
- **Audio format**: WAV recommended, but MP3, M4A, and other formats are supported
- **Duration**: 5-24 seconds of clear speech
- **Quality**: Clean recording with minimal background noise

## Parameter Tuning

### Controlling Voice Accent

This model offers excellent control over accents and pronunciation:

- **For preserving the speaker's accent**:
  - Set pronunciation_strength (p_w) to a lower value (1.0-1.5)
  - This is useful for cross-lingual TTS where you want to preserve the accent

- **For standard pronunciation**:
  - Set pronunciation_strength (p_w) to a higher value (2.5-4.0)
  - This helps produce more standard pronunciation regardless of the source accent

- **For emotional or expressive speech**:
  - Increase the voice_similarity (t_w) parameter (2.0-5.0) 
  - Keep pronunciation_strength (p_w) at a moderate level (1.5-2.5)

### Recommended Parameter Combinations

| Use Case | p_w (pronunciation_strength) | t_w (voice_similarity) |
|----------|------------------------------|------------------------|
| Standard TTS | 2.0 | 3.0 |
| Preserve Accent | 1.0-1.5 | 3.0-5.0 |
| Cross-lingual (standard) | 3.0-4.0 | 3.0-5.0 |
| Emotional Speech | 1.5-2.5 | 3.0-5.0 |
| Noisy Reference Audio | 3.0-5.0 | 3.0-5.0 |

## Nodes

This extension provides three main nodes:

### 1. MegaTTS3 (Advanced)

Full-featured TTS node with complete parameter control.

**Inputs:**
- `input_text` - Text to convert to speech
- `language` - Language selection (en: English, zh: Chinese)
- `generation_quality` - Controls the number of diffusion steps (higher = better quality but slower)
- `pronunciation_strength` (p_w) - Controls pronunciation accuracy (higher values produce more standard pronunciation)
- `voice_similarity` (t_w) - Controls similarity to reference voice (higher values produce speech more similar to reference)
- `reference_voice` - Reference voice file from Voices folder

**Outputs:**
- `AUDIO` - Generated audio in WAV format
- `LATENT` - Audio latent representation for further processing

### 2. MegaTTS3 (Simple)

Simplified TTS node with default parameters for quick usage.

**Inputs:**
- `input_text` - Text to convert to speech
- `language` - Language selection (en: English, zh: Chinese)
- `reference_voice` - Reference voice file from Voices folder

**Outputs:**
- `AUDIO` - Generated audio in WAV format


Utility node to free GPU memory after TTS processing.

## Parameter Descriptions

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| **generation_quality** | Controls the number of diffusion steps. Higher values produce better quality but increase generation time. | Default: 10. Range: 1-50. For quick tests: 1-5, for final output: 15-30. |
| **pronunciation_strength** (p_w) | Controls how closely the output follows standard pronunciation. | Default: 2.0. Range: 1.0-5.0. For accent preservation: 1.0-1.5, for standard pronunciation: 2.5-4.0. |
| **voice_similarity** (t_w) | Controls how similar the output is to the reference voice. | Default: 3.0. Range: 1.0-5.0. For more expressive output with preserved voice characteristics: 3.0-5.0. |

## Voice Cloning

### Adding Reference Voices

1. Place your voice WAV files in the `Voices` folder
2. Each voice requires two files:
   - `voice_name.wav` - Voice sample file (24kHz sample rate recommended, 5-10 seconds of clear speech)
   - `voice_name.npy` - Corresponding voice feature file (generated automatically if voice extraction is enabled)

### How to Clone a Voice

1. Add your sample WAV file to the `Voices` folder
2. The first time you select the voice, the system will extract feature files and save them
3. Select your voice in the node's "reference_voice" dropdown
4. Adjust the "voice_similarity" parameter to control the intensity of voice cloning:
   - Lower values (1.0-2.0): More natural but less similar to reference
   - Higher values (3.0-5.0): More similar to reference but potentially less natural

## Advanced Usage

### Cross-Language Voice Cloning

For cloning a voice across languages (e.g., making an English speaker speak Chinese):

1. Use a clean voice sample in the original language
2. Set language to the target language (e.g., "zh" for Chinese)
3. Increase the pronunciation_strength (p_w) parameter (3.0-4.0) for more standard pronunciation
4. Set voice_similarity (t_w) parameter higher (3.0-5.0) to maintain voice characteristics

### Handling Accents

- For preserving accents: Lower pronunciation_strength (p_w) value (1.0-1.5)
- For standard pronunciation: Higher pronunciation_strength (p_w) value (2.5-4.0)

## Credits

- Original MegaTTS3 model by [ByteDance](https://github.com/bytedance/MegaTTS3)
- MegaTTS3 Hugging Face model: [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)

## License
GPL-3.0 License

## References

- [ByteDance MegaTTS3 GitHub Repository](https://github.com/bytedance/MegaTTS3)
- [ByteDance MegaTTS3 Hugging Face Model](https://huggingface.co/ByteDance/MegaTTS3)
- Original papers:
  - ["Sparse Alignment Enhanced Latent Diffusion Transformer for Zero-Shot Speech Synthesis"](https://arxiv.org/abs/2502.18924)
  - ["Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling"](https://arxiv.org/abs/2408.16532) 
