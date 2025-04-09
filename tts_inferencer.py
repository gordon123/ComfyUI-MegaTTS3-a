import json
import os
import librosa
import numpy as np
import torch
import pyloudnorm as pyln
import folder_paths
import soundfile as sf
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pydub import AudioSegment
AudioSegment.ffprobe = "/usr/bin/ffprobe"

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
from tts.frontend_function import g2p, align, make_dur_prompt, dur_pred, prepare_inputs_for_dit
from tts.utils.audio_utils.io import save_wav, to_wav_bytes, convert_to_wav_bytes, combine_audio_segments
from tts.utils.commons.ckpt_utils import load_ckpt
from tts.utils.commons.hparams import set_hparams, hparams
from tts.utils.text_utils.text_encoder import TokenTextEncoder
from tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

def convert_audio_format(audio_bytes, target_sr=24000, max_duration=10.0):
    try:
        from pydub import AudioSegment
        
        temp_in = os.path.join(model_path, "temp_in.audio")
        temp_out = os.path.join(model_path, "temp_out.wav")
        
        with open(temp_in, 'wb') as f:
            f.write(audio_bytes)
        
        audio = AudioSegment.from_file(temp_in)
        
        if max_duration > 0:
            max_ms = int(max_duration * 1000)
            if len(audio) > max_ms:
                audio = audio[:max_ms]
        
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
        
        audio.export(temp_out, format="wav")
        
        with open(temp_out, 'rb') as f:
            wav_bytes = f.read()
            
        os.remove(temp_in)
        os.remove(temp_out)
        
        return wav_bytes
    
    except Exception as e:
        print(f"Audio conversion error: {e}")
        
        try:
            wav_bytes = convert_to_wav_bytes(audio_bytes)
            wav_bytes = wav_bytes.getvalue()
            
            y, sr = librosa.load(io.BytesIO(wav_bytes), sr=target_sr)
            
            if max_duration > 0 and len(y) > max_duration * target_sr:
                y = y[:int(max_duration * target_sr)]
            
            out_buffer = io.BytesIO()
            sf.write(out_buffer, y, target_sr, format='WAV')
            out_buffer.seek(0)
            
            return out_buffer.read()
        
        except Exception as e2:
            print(f"Backup audio conversion also failed: {e2}")
            raise ValueError(f"Could not convert audio format: {e} -> {e2}")

class TTSInferencer:
    def __init__(self, device=None, ckpt_root=os.path.join(model_path, "MegaTTS3"),
                 dit_exp_name='diffusion_transformer', frontend_exp_name='aligner_lm',
                 wavvae_exp_name='wavvae', dur_ckpt_path='duration_lm',
                 g2p_exp_name='g2p', precision=torch.float16, **kwargs):
        self.sr = 24000
        self.fm = 8
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.precision = precision
        
        self.dit_exp_name = os.path.join(ckpt_root, dit_exp_name)
        self.frontend_exp_name = os.path.join(ckpt_root, frontend_exp_name)
        self.wavvae_exp_name = os.path.join(ckpt_root, wavvae_exp_name)
        self.dur_exp_name = os.path.join(ckpt_root, dur_ckpt_path)
        self.g2p_exp_name = os.path.join(ckpt_root, g2p_exp_name)
        
        with open(os.devnull, 'w') as f, redirect_stdout(f), redirect_stderr(f):
            self.build_model(self.device)
        
        self.loudness_meter = pyln.Meter(self.sr)
        
    def clean(self):
        import gc
        self.dur_model = self.dit = self.g2p_model = self.wavvae = None
        gc.collect()
        torch.cuda.empty_cache()

    def build_model(self, device):
        set_hparams(exp_name=self.dit_exp_name, print_hparams=False)
        self._load_linguistic_dict()
        self._init_duration_model(device)
        self._init_diffusion_model(device)
        self._init_aligner_model(device)
        self._init_g2p_model(device)
        self._init_wavvae_model(device)

    def _load_linguistic_dict(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ling_dict = json.load(open(f"{current_dir}/tts/utils/text_utils/dict.json", encoding='utf-8-sig'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') 
                         for k in ['phone', 'tone']}
        self.token_encoder = self.ling_dict['phone']

    def _init_duration_model(self, device):
        from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor
        
        hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
        hp_dur_model['frames_multiple'] = hparams['frames_multiple']
        
        ph_dict_size = len(self.token_encoder)
        self.dur_model = ARDurPredictor(
            hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
            hp_dur_model['dur_model_layers'], ph_dict_size, hp_dur_model['dur_code_size'],
            use_rot_embed=hp_dur_model.get('use_rot_embed', False))
        self.length_regulator = LengthRegulator()
        
        load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
        self.dur_model.eval().to(device)

    def _init_diffusion_model(self, device):
        from tts.modules.llm_dit.dit import Diffusion
        self.dit = Diffusion()
        load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False)
        self.dit.eval().to(device)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1

    def _init_aligner_model(self, device):
        from tts.modules.aligner.whisper_small import Whisper
        self.aligner_lm = Whisper()
        load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
        self.aligner_lm.eval().to(device)
        self.kv_cache = None
        self.hooks = None

    def _init_g2p_model(self, device):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        g2p_tokenizer = AutoTokenizer.from_pretrained(self.g2p_exp_name, padding_side="right")
        g2p_tokenizer.padding_side = "right"
        self.g2p_model = AutoModelForCausalLM.from_pretrained(self.g2p_exp_name).eval().to(device)
        self.g2p_tokenizer = g2p_tokenizer
        self.speech_start_idx = g2p_tokenizer.encode('<Reserved_TTS_0>')[0]

    def _init_wavvae_model(self, device):
        from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3
        
        self.hp_wavvae = hp_wavvae = set_hparams(f'{self.wavvae_exp_name}/config.yaml', global_hparams=False)
        self.wavvae = WavVAE_V3(hparams=hp_wavvae)
        
        if os.path.exists(f'{self.wavvae_exp_name}/model_only_last.ckpt'):
            load_ckpt(self.wavvae, f'{self.wavvae_exp_name}/model_only_last.ckpt', 'model_gen', strict=True)
            self.has_vae_encoder = True
        else:
            load_ckpt(self.wavvae, f'{self.wavvae_exp_name}/decoder.ckpt', 'model_gen', strict=False)
            self.has_vae_encoder = False
            
        self.wavvae.eval().to(device)
        self.vae_stride = hp_wavvae.get('vae_stride', 4)
        self.hop_size = hp_wavvae.get('hop_size', 4)

    def preprocess(self, audio_bytes, latent_file=None, **kwargs):
        wav_bytes = convert_audio_format(audio_bytes, target_sr=self.sr, max_duration=10.0)
        wav, _ = librosa.load(io.BytesIO(wav_bytes), sr=self.sr)
        
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
        
        self.loudness_prompt = self.loudness_meter.integrated_loudness(wav.astype(float))
        ph_ref, tone_ref, mel2ph_ref = align(self, wav)

        with torch.inference_mode():
            if latent_file is not None:
                vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
            else:
                latent_size = mel2ph_ref.size(1)//4
                vae_latent = torch.zeros((1, latent_size, 32)).to(self.device)

            incremental_state_dur_prompt, ctx_dur_tokens = make_dur_prompt(self, mel2ph_ref, ph_ref, tone_ref)

        return {
            'ph_ref': ph_ref,
            'tone_ref': tone_ref,
            'mel2ph_ref': mel2ph_ref,
            'vae_latent': vae_latent,
            'incremental_state_dur_prompt': incremental_state_dur_prompt,
            'ctx_dur_tokens': ctx_dur_tokens,
        }

    def forward(self, resource_context, input_text, language_type, time_step, p_w, t_w, 
                dur_disturb=0.1, dur_alpha=1.0, **kwargs):
        device = self.device
        
        ph_ref = resource_context['ph_ref'].to(device)
        tone_ref = resource_context['tone_ref'].to(device)
        mel2ph_ref = resource_context['mel2ph_ref'].to(device)
        vae_latent = resource_context['vae_latent'].to(device)
        ctx_dur_tokens = resource_context['ctx_dur_tokens'].to(device)
        incremental_state_dur_prompt = resource_context['incremental_state_dur_prompt']

        with torch.inference_mode():
            wav_pred_ = []
            text_segs = (chunk_text_english(input_text, max_chars=130) if language_type == 'en' 
                        else chunk_text_chinese(input_text, limit=60))

            for seg_i, text in enumerate(text_segs):
                ph_pred, tone_pred = g2p(self, text)
                mel2ph_pred = dur_pred(self, ctx_dur_tokens, incremental_state_dur_prompt, 
                                     ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha,
                                     is_first=seg_i==0, is_final=seg_i==len(text_segs)-1)
                
                inputs = prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, 
                                              tone_ref, ph_pred, tone_pred, vae_latent)
                with torch.amp.autocast('cuda', dtype=self.precision, enabled=True):
                    x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()
                
                x[:, :vae_latent.size(1)] = vae_latent
                wav_pred = self.wavvae.decode(x)[0,0].to(torch.float32)
                
                wav_pred = wav_pred[vae_latent.size(1)*self.vae_stride*self.hop_size:].cpu().numpy()
                loudness_pred = self.loudness_meter.integrated_loudness(wav_pred.astype(float))
                wav_pred = pyln.normalize.loudness(wav_pred, loudness_pred, self.loudness_prompt)
                if np.abs(wav_pred).max() >= 1:
                    wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95

                wav_pred_.append(wav_pred)

            wav_pred = combine_audio_segments(wav_pred_, sr=self.sr).astype(np.float32)
            waveform = torch.tensor(wav_pred).unsqueeze(0).unsqueeze(0)

            return {"waveform": waveform, "sample_rate": self.sr} 
