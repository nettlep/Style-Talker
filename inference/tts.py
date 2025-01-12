import os
import sys
import yaml
import librosa
import numpy as np
from munch import Munch
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

import phonemizer
from nltk.tokenize import word_tokenize

# StyleTTS2's utils
from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import (
    DiffusionSampler, ADPM2Sampler, KarrasSchedule
)


class StyleTTS2(nn.Module):

    def __init__(self,
        ckpt_root,
        code_root,
        espeak_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.ckpt_root = ckpt_root
        self.code_root = code_root
        self.device = device

        # Phonemizer
        if espeak_path != None:
            phonemizer.backend.EspeakBackend.set_library(espeak_path)
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True,  with_stress=True
        )

        self.textcleaner = TextCleaner()

        # Feature extractor
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

        #### pretrained models ####

        # Config
        config = yaml.safe_load(open(
                os.path.join(
                    code_root, 'Configs/config_libritts.yml'
                )
            )
        )

        # ASR model
        ASR_config = os.path.join(code_root, config.get('ASR_config', False))
        ASR_path = os.path.join(code_root, config.get('ASR_path', False))
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # F0 model
        F0_path = os.path.join(code_root, config.get('F0_path', False))
        pitch_extractor = load_F0_models(F0_path)

        # BERT model
        from Utils.PLBERT.util import load_plbert
        BERT_path = os.path.join(code_root, config.get('PLBERT_dir', False))
        plbert = load_plbert(BERT_path)

        # Load all weights
        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(device) for key in self.model]
        params_whole = torch.load(ckpt_root, map_location=device)
        params = params_whole['net']
        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)

        _ = [self.model[key].eval() for key in self.model]

        # Diffusion
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


    def preprocess(self, wave, mean=-4, std=4):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor


    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)


    def forward(self, text, ref_s, alpha=0.3, beta=0.0, diffusion_steps=5, embedding_scale=1):
        text = replace_text(text)
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device), 
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s, # reference from the same speaker as the embedding
                num_steps=diffusion_steps
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(
                d_en, s, input_lengths, text_mask
            )

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
        # weird pulse at the end of the model, need to be fixed later
        return out.squeeze().cpu().numpy()[..., :-50]



def replace_text(text):
    replace_labels = [
        '303',
        '504',
        'S1',
        'S2',
        'S3',
        's2',
        'c',
        'high-pitched',
        'inaudible',
        'indistinct',
        'k',
        'music',
        'outro',
        'outro music',
        's1',
        'sad music',
        'singing',
        'speaking in foreign language',
        'speaks in foreign language',
        'sped up',
        'static',
        'upbeat music',
        'whispering',
        'speaking in Linguan',
        'vocalizing',
        'faint singing',
        'mumbles'
    ]
    for r in replace_labels:
        text = text.replace('(' + r + ')', '')
        
    token_replace = {'airplane engine': '^',
     'applause': '↓',
     'audience applause': '↓',
     'audience laughs': '↑',
     'background noise': ' ',
     'beep': '«',
     'bell dings': '¿',
     'chicken clucking': '↗',
     'clears throat': '↘',
     'cough': 'ǂ',
     'coughing': 'ǂ',
     'coughs': 'ǂ',
     'ding': '¿',
     'dog barking': 'ǃ',
    'dog barks':  'ǃ',
     'finger snaps': 'ǀ',
     'gagging': 'ǁ',
     'gasps': 'ǁ',
     'groans': 'ǁ',
     'laughing': '¡',
     'laughs': '¡',
     'laughter': '¡',
     'makes noise': ' ',
     'record scratch': ' ',
     'sighs': 'ʡ',
     'slapping': 'ʕ',
     'squelch': 'ʢ',
     'stammers': 'ʄ',
     'stuttering': 'ʄ',
     'stutters': 'ʄ'}
    
    for k, v in token_replace.items():
        text = text.replace('(' + k + ')', v)
        
    token_replace = {
        'Applause': '↓',
        'BLANK_AUDIO': ' ',
        'MUSIC PLAYING': '',
        'S3': '',
        'laughs': '¡',
        'laughter': '¡',
        'music': '',
        'outro': '',
        's1': '',
        'male announcer': '',
        'singing': ''
    }
    
    for k, v in token_replace.items():
        text = text.replace('[' + k + ']', v)
        
    return text
