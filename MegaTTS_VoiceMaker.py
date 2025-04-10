import os
import io
import numpy as np
import torch
import librosa
import soundfile as sf
from .tts_inferencer import TTSInferencer
from .MegaTTS_utils import initialize
from .AILab_MegaTTS import MegaTTS3

class MegaTTS_VoiceMaker:
    infer_instance_cache = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_in": ("AUDIO", {"tooltip": "Input audio to be converted."}),
                "voice_name": ("STRING", {"default": "my_voice", "tooltip": "Name of the voice to be used for conversion."}),
                "path": ("STRING", {"default": "", "placeholder": "Voices", "tooltip": "Directory path where the voice files will be saved. If empty, will use default 'Voices' folder."}),
                "trim_silence": ("BOOLEAN", {"default": True, "tooltip": "Whether to trim silence from the audio."}),
                "normalize_volume": ("BOOLEAN", {"default": True, "tooltip": "Whether to normalize the volume of the audio."}),
                "max_duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 0.5, "tooltip": "Maximum duration of the audio in seconds."})
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio_out", "voice_path",)
    FUNCTION = "convert_voice"
    CATEGORY = "🧪AILab/🔊Audio"

    def convert_voice(self, audio_in, voice_name, path="", trim_silence=True, normalize_volume=True, max_duration=10.0):
        # Initialize the model if not already done.
        if not MegaTTS3.initialization_done:
            initialize()
            MegaTTS3.initialization_done = True
            
        if MegaTTS_VoiceMaker.infer_instance_cache is not None:
            infer_instance = MegaTTS_VoiceMaker.infer_instance_cache
        else:
            infer_instance = MegaTTS_VoiceMaker.infer_instance_cache = TTSInferencer()
        
        if audio_in is None or not isinstance(audio_in, dict) or 'waveform' not in audio_in:
            return ({"waveform": torch.zeros(1, 1), "sample_rate": 24000}, "No input audio provided")
        
        waveform = audio_in['waveform']
        sample_rate = audio_in.get('sample_rate', 44100)
        
        # Debug: print input waveform shape and sample rate
        print("DEBUG: Input waveform shape:", waveform.shape)
        print("DEBUG: Input sample_rate:", sample_rate)
        
        if not torch.is_tensor(waveform):
            return (audio_in, "Error: Waveform must be a tensor")
        
        samples = waveform.cpu().numpy()
        print("DEBUG: Converted samples shape:", samples.shape)
        
        if len(samples.shape) > 1:
            samples = samples.squeeze()
            print("DEBUG: Squeezed samples shape:", samples.shape)
        samples = samples.astype(np.float32)
        print("DEBUG: Samples dtype after conversion:", samples.dtype)
        print("DEBUG: Samples min, max before processing:", np.min(samples), np.max(samples))
        
        if sample_rate != infer_instance.sr:
            print("DEBUG: Resampling from", sample_rate, "to", infer_instance.sr)
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=infer_instance.sr)
            print("DEBUG: Samples shape after resampling:", samples.shape)
        
        max_samples = int(max_duration * infer_instance.sr)
        if len(samples) > max_samples:
            print("DEBUG: Truncating samples from", len(samples), "to", max_samples)
            samples = samples[:max_samples]
        
        if trim_silence:
            print("DEBUG: Trimming silence")
            samples, _ = librosa.effects.trim(samples, top_db=30)
            print("DEBUG: Samples shape after trimming:", samples.shape)
        
        if normalize_volume:
            print("DEBUG: Normalizing volume")
            samples = librosa.util.normalize(samples)
            print("DEBUG: Samples min, max after normalization:", np.min(samples), np.max(samples))
        
        # Set up the directory for saving reference files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        voices_dir = os.path.join(current_dir, "Voices" if path == "" else path)
        os.makedirs(voices_dir, exist_ok=True)
        
        output_wav_path = os.path.join(voices_dir, f"{voice_name}.wav")
        output_npy_path = os.path.join(voices_dir, f"{voice_name}.npy")
        
        print("DEBUG: Writing WAV file to", output_wav_path)
        sf.write(output_wav_path, samples, infer_instance.sr)
        
        wav_io = io.BytesIO()
        print("DEBUG: Writing audio to BytesIO with explicit format='WAV'")
        sf.write(wav_io, samples, infer_instance.sr, format='WAV')
        voice_data = wav_io.getvalue()
        print("DEBUG: Length of audio bytes in BytesIO:", len(voice_data))
        
        try:
            print("DEBUG: Running infer_instance.preprocess")
            resource_context = infer_instance.preprocess(
                voice_data, 
                use_encoder_mode=True,
                topk_dur=1
            )
        except Exception as e:
            print("DEBUG: Exception in preprocess:", e)
            return (audio_in, f"Error in preprocessing: {e}")
        
        vae_latent = resource_context.get('vae_latent')
        if vae_latent is None:
            print("DEBUG: No 'vae_latent' found in resource_context")
            return (audio_in, "Error: No latent representation extracted")
        
        print("DEBUG: vae_latent shape:", vae_latent.shape)
        vae_latent_np = vae_latent.cpu().numpy()
        print("DEBUG: vae_latent max value:", np.max(vae_latent_np))
        np.save(output_npy_path, vae_latent_np)
        print("DEBUG: Saved latent features to", output_npy_path)
        
        status = f"✅ Successfully processed and saved reference voice '{voice_name}'\n"
        status += f"• WAV: {output_wav_path}\n"
        status += f"• NPY: {output_npy_path}\n"
        status += f"• Duration: {len(samples)/infer_instance.sr:.2f} seconds\n"
        status += f"• Sample rate: {infer_instance.sr}Hz\n"
        status += f"• Feature shape: {vae_latent_np.shape}"
        
        return (audio_in, status)

NODE_CLASS_MAPPINGS = {
    "MegaTTS_VoiceMaker": MegaTTS_VoiceMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaTTS_VoiceMaker": "Voice Maker for MegaTTS3"
}
