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
        # 确保模型已下载
        if not getattr(MegaTTS3, 'initialization_done', False):
            initialize()
            MegaTTS3.initialization_done = True
            
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
        if MegaTTS_VoiceMaker.infer_instance_cache is not None:
            infer_instance = MegaTTS_VoiceMaker.infer_instance_cache
        else:
            infer_instance = MegaTTS_VoiceMaker.infer_instance_cache = TTSInferencer()
        
        if audio_in is None or not isinstance(audio_in, dict) or 'waveform' not in audio_in:
            return ({"waveform": torch.zeros(1, 1), "sample_rate": 24000}, "No input audio provided")
        
        waveform = audio_in['waveform']
        sample_rate = audio_in.get('sample_rate', 44100)
        
        if not torch.is_tensor(waveform):
            return (audio_in, "Error: Waveform must be a tensor")
        
        samples = waveform.cpu().numpy()
        if len(samples.shape) > 1:
            samples = samples.squeeze()
        samples = samples.astype(np.float32)
        
        if sample_rate != infer_instance.sr:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=infer_instance.sr)
        
        max_samples = int(max_duration * infer_instance.sr)
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        
        if trim_silence:
            samples, _ = librosa.effects.trim(samples, top_db=30)
        
        if normalize_volume:
            samples = librosa.util.normalize(samples)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        voices_dir = os.path.join(current_dir, "Voices" if path == "" else path)
        os.makedirs(voices_dir, exist_ok=True)
        
        output_wav_path = os.path.join(voices_dir, f"{voice_name}.wav")
        output_npy_path = os.path.join(voices_dir, f"{voice_name}.npy")
        
        sf.write(output_wav_path, samples, infer_instance.sr)
        
        wav_io = io.BytesIO()
        sf.write(wav_io, samples, infer_instance.sr, format='WAV')
        voice_data = wav_io.getvalue()
        
        resource_context = infer_instance.preprocess(
            voice_data, 
            use_encoder_mode=True,
            topk_dur=1
        )
        
        vae_latent = resource_context['vae_latent'].cpu().numpy()
        np.save(output_npy_path, vae_latent)
        
        status = f"✅ Successfully processed and saved reference voice '{voice_name}'\n"
        status += f"• WAV: {output_wav_path}\n"
        status += f"• NPY: {output_npy_path}\n"
        status += f"• Duration: {len(samples)/infer_instance.sr:.2f} seconds\n"
        status += f"• Sample rate: {infer_instance.sr}Hz\n"
        status += f"• Feature shape: {vae_latent.shape}"
        
        return (audio_in, status)

NODE_CLASS_MAPPINGS = {
    "MegaTTS_VoiceMaker": MegaTTS_VoiceMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaTTS_VoiceMaker": "Voice Maker for MegaTTS3"
} 