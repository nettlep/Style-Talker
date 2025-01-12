import torch
from torch import nn
import torchaudio

import librosa
import numpy as np
from transformers import pipeline
from typing import List, Optional, Union

from inference.tts import StyleTTS2
from inference.audiollm import StyleQwen

from data.dataset import preprocess


spks_template = "spk_{spk_id1}: STYLE: <|extra_123|> spk_{spk_id2}: STYLE: <|extra_123|>"
history_template = "spk_{spk_id}: STYLE: <|extra_123|> TEXT: '{text}'"
input_template = "Audio 1:<audio>{audio_path}</audio>\nThis is the voice of the {last_spk} last speaking. There is a conversation among {speakers_and_styles}. Here is some context: \n\n{hist}\n\nTry to recognize what {last_spk} just said from the audio, and generate the style and text of the next speaker {next_spk}. Be creative and avoid repeated words and sentences. STYLE: <|extra_124|> TEXT: "

N_ROUND = 3 # fixed

class StyleTalker(nn.Module):

    def __init__(self,
        tts_ckpt_root,
        audiollm_ckpt_root,
        tts_code_root,
        audiollm_kwargs,
        asr_model=None,
        espeak_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.audiollm = StyleQwen(
            ckpt_root=audiollm_ckpt_root,
            model_name='Qwen/Qwen-Audio-Chat',
            device=device,
            **audiollm_kwargs,
        )
        self.audiollm_bf16 = audiollm_kwargs['bf16']
        print('Initialized adapted & finetuned QwenAudio for dialog understanding.')
        
        self.tts = StyleTTS2(
            ckpt_root=tts_ckpt_root,
            code_root=tts_code_root,
            espeak_path=espeak_path,
            device=device
        )
        print('Initialized finetuned StyleTTS2 for dialog generation.')

        if asr_model != None:
            self.asr = pipeline(
                'automatic-speech-recognition', 
                model=asr_model, device=device
            )
            print(f'Initialized {asr_model} for offline speech recognition.')

        self.history_texts = []
        self.history_styles = []
        self.device = device


    @torch.no_grad()
    def transcribe(self, wav_file):
        return self.asr(
            wav_file, 
            generate_kwargs={"language": "english"}
        )['text'].strip()


    @torch.no_grad()
    def compute_style(self, wav_file):

        def preprocess(wav, mean=-4, std=4):
            wav_tensor = torch.from_numpy(wav).float()
            mel_tensor = self.tts.to_mel(wav_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
            return mel_tensor

        wav, sr = librosa.load(wav_file, sr=24000)
        audio, index = librosa.effects.trim(wav, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.tts.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.tts.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1).squeeze(0)

        
    @torch.no_grad()
    def forward(
        self,
        latest_speech: str,
        history_texts: Optional[List[str]] = None,
        history_styles: Optional[List[torch.Tensor]] = None,
        history_speeches: Optional[List[str]] = None,
        # Optional; Pass in as references
        reference_styles: Optional[List[torch.Tensor]] = None,
        speaker_identity: torch.Tensor = None,
        first_spk_id: Optional[int] = 1,
    ): 
        '''
        One of (history_texts, history_styles) or history_speeches must be given
        '''

        if (history_texts is not None and history_styles is not None) or history_speeches is not None:
            if history_speeches is not None:
                # Ensure class has `asr` attribute
                assert hasattr(self, 'asr'), \
                "If history_speeches is provided, the class must have `asr_model` to transcribe history_speeches"

                # Transcribe history speeches
                history_texts = [
                    self.transcribe(history_speech)
                    for history_speech in history_speeches
                ]

                # Compute history styles
                history_styles = [
                    self.compute_style(history_speech)
                    for history_speech in history_speeches
                ]

        else:
            raise ValueError("Either (history_texts and history_styles) must be provided, or history_speeches must be provided")

        # The current model uses three N_ROUND=3 rounds of conversation;
        # The last round is latest_speech and the previous two rounds 
        # texts and styles, preprocessed from speeches
        assert len(history_texts) == len(history_styles) == N_ROUND - 1
        assert first_spk_id in [0, 1]
        spk_ids = [(first_spk_id + i) % 2 for i in range(0, N_ROUND+1)] # (0, 1, 0, 1)

        # List two speakers in the conversation by the order of first and second
        speakers_and_styles = spks_template.format(
            spk_id1=spk_ids[0], spk_id2=spk_ids[1] 
        )

        if (history_texts is not None and history_styles is not None):
            # Append conversation history
            histories = ''
            for history_text, history_style, spk_id in zip(history_texts, history_styles, spk_ids[:-2]):
                if histories != '':
                    histories += '\n'
                histories += history_template.format(
                    spk_id=spk_id, text=history_text
                )

            prompt = input_template.format(
                audio_path=latest_speech, 
                speakers_and_styles=speakers_and_styles,
                last_spk='spk_'+str(spk_ids[-2]),
                hist=histories,
                next_spk='spk_'+str(spk_ids[-1]),
            )

            # If no reference_styles given, sample a style
            # for each speaker from history_styles
            if reference_styles == None:
                reference_styles = [history_styles[0], history_styles[1]]
            else:
                assert len(reference_styles) == 2

            input_styles = torch.stack(
                reference_styles+history_styles 
            )

            # If no speaker_identity is given, reuse the first 128 dims of style 
            # from the last round of conversation for the speaker to be generated
            if speaker_identity == None:
                speaker_identity = history_styles[-1][:128].unsqueeze(0) # (1, 128)

            inputs = preprocess(
                [[
                    {
                        "from": "user",
                        "value": prompt
                    },
                ]], 
                self.audiollm.tokenizer, 
                self.audiollm.max_window_size
            )

            if inputs == None:
                raise ValueError(f'Maximum token length {str(self.audiollm.max_window_size)} exceeded!')

            inputs = dict(
                input_ids=inputs["input_ids"][0],
                labels=inputs["labels"][0],
                attention_mask=inputs["attention_mask"][0],
                audio_info=inputs["audio_info"][0],
                in_styles=input_styles,
                prompt=prompt
            )

            # Forward Audio LLM
            if self.audiollm_bf16:
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    est_text, est_prosody = self.audiollm(inputs)
            else:
                est_text, est_prosody = self.audiollm(inputs)


            # gt_style = torch.load('samples/dailytalk/1/r6.pt').unsqueeze(0).cuda()        

            est_style = torch.cat(
                [speaker_identity.to(est_prosody.device), 
                est_prosody[:, 128:]], 
                dim=1
            )

            # Forward TTS
            est_wav = self.tts(text=est_text, ref_s=est_style.float())

            return {'text': est_text, 'audio': est_wav}