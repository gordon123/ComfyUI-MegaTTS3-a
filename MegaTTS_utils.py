import json
import os
import sys
import urllib.request
import traceback
from tqdm import tqdm
import folder_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

MODELS_DIR = folder_paths.models_dir
TTS_MODEL_PATH = os.path.join(MODELS_DIR, "TTS")
MEGATTS3_MODEL_PATH = os.path.join(TTS_MODEL_PATH, "MegaTTS3")

MODEL_BASE_URL = "https://huggingface.co/ByteDance/MegaTTS3/resolve/main"

MODEL_FILES = [
    "diffusion_transformer/config.yaml",
    "diffusion_transformer/model_only_last.ckpt",
    "wavvae/config.yaml",
    "wavvae/decoder.ckpt",
    "duration_lm/config.yaml",
    "duration_lm/model_only_last.ckpt",
    "aligner_lm/config.yaml",
    "aligner_lm/model_only_last.ckpt",
    "g2p/config.json",
    "g2p/model.safetensors",
    "g2p/generation_config.json", 
    "g2p/tokenizer_config.json",
    "g2p/special_tokens_map.json",
    "g2p/tokenizer.json",
    "g2p/vocab.json",
    "g2p/merges.txt"
]

CORE_FILES = [
    os.path.join(MEGATTS3_MODEL_PATH, "diffusion_transformer", "model_only_last.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "wavvae", "decoder.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "duration_lm", "model_only_last.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "aligner_lm", "model_only_last.ckpt"),
    os.path.join(MEGATTS3_MODEL_PATH, "g2p", "model.safetensors")
]

def get_voice_samples():
    voice_samples_dir = os.path.join(current_dir, "Voices")
    os.makedirs(voice_samples_dir, exist_ok=True)
    
    return [f for f in os.listdir(voice_samples_dir) if f.endswith('.wav')]

def get_voice_path(voice_name):
    voice_path = os.path.join(current_dir, "Voices", voice_name)
    return voice_path

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading: {os.path.basename(destination)}") as t:
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Download error {url}: {str(e)}")
        return False

def check_and_download_models():
    if getattr(check_and_download_models, 'completed', False):
        return True
        
    os.makedirs(TTS_MODEL_PATH, exist_ok=True)
    os.makedirs(MEGATTS3_MODEL_PATH, exist_ok=True)
    
    voice_dir = os.path.join(current_dir, "Voices")
    os.makedirs(voice_dir, exist_ok=True)
    
    missing_files = [f for f in CORE_FILES if not os.path.exists(f)]
    
    if not missing_files:
        check_and_download_models.completed = True
        return True
        
    print("Starting to download required model files...")
    success = True
    
    for file_path in MODEL_FILES:
        dest_path = os.path.join(MEGATTS3_MODEL_PATH, file_path)
        
        if os.path.exists(dest_path):
            continue
            
        download_url = f"{MODEL_BASE_URL}/{file_path}"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        if not download_file(download_url, dest_path):
            success = False
            print(f"Download failed: {file_path}")
    
    if success:
        check_and_download_models.completed = True
        print("All model files downloaded successfully.")
    else:
        print("Some model files could not be downloaded. Please check your network connection and try again.")
    
    return success

initialization_completed = False

def initialize():
    global initialization_completed
    if initialization_completed:
        return True
        
    try:
        print("Initializing MegaTTS for the first time...")
        
        if check_and_download_models():
            print(f"MegaTTS model is ready: {MEGATTS3_MODEL_PATH}")
            
            samples = get_voice_samples()
            if samples:
                print(f"Available voice samples: {len(samples)}")
            else:
                print("No voice samples found. Please add .wav files in the Voices directory.")
                
            initialization_completed = True
            return True
        else:
            print("Model initialization failed. Please check your network connection and try again.")
            return False
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        traceback.print_exc()
        return False