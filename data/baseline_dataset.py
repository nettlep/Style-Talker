import os
import re
import json
import pickle
import random
import numpy as np
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

input_template = "There is a conversation among{speakers_and_styles}. Here is some context: \n\n{hist}\n\nBe creative and avoid repeated words and sentences. Generate the next response of {next_spk} TEXT: "

import transformers
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict, Optional, Sequence, List

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DataCollatorQwen(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels, attention_mask= tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "attention_mask"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_TOKEN_ID)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class TextConversations(Dataset):

    def __init__(
        self, 
        data_root, 
        tokenizer: transformers.PreTrainedTokenizer, 
        max_len: int=2048,
        max_turn: int=99999,
        tokenize: bool=True
    ):
        super(TextConversations, self).__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_turn = max_turn
        self.tokenize = tokenize
        self.cached_data_dict = {}
        
        self.style_in_id = 151769
        self.style_out_id = 151770
        
        pattern = re.compile(r'^\d+$')
        
        self.conv_dirs = []
        self.styles = []
        self.texts = []
        self.audio_paths = []
        self.spk_styles = []
        self.crops = {}
        self.ABABs = {}
        
        # Find all conversations
        for conv in os.listdir(data_root):
            if pattern.match(conv):
                conv_dir = os.path.join(data_root, conv)
                self.conv_dirs.append(conv_dir)
        self.conv_dirs.sort()
        
        for i, conv_dir in enumerate(self.conv_dirs):
            
            # Load all sentence styles
            styles = torch.tensor(np.load(
                os.path.join(conv_dir, 'styles.npy')
            ))
            self.styles.append(styles)
            
            # Load all styles from all speakers
            with open(os.path.join(conv_dir, 'spk_styles.pkl'), 'rb') as file:
                spk_styles = pickle.load(file)
            self.spk_styles.append(spk_styles)

            # Load all texts from the conv
            with open(os.path.join(conv_dir, 'texts.txt'), 'r') as file:
                texts = file.readlines()
                texts = [line.strip() for line in texts]
            self.texts.append(texts)

            # Load all audio paths from the conv
            with open(os.path.join(conv_dir, 'audio_paths.txt'), 'r') as file:
                paths = file.readlines()
                paths = [line.strip() for line in paths]
            self.audio_paths.append(paths)
            
            assert len(styles) == len(texts) == len(paths)

            crop_pkl = os.path.join(conv_dir, 'crops.pkl')
            if os.path.exists(crop_pkl):
                with open(crop_pkl, 'rb') as file:
                    self.crops[i] = pickle.load(file)
            else:
                self.crops[i] = []

            ABAB_pkl = os.path.join(conv_dir, 'ABAB.pkl')
            if os.path.exists(ABAB_pkl):
                with open(ABAB_pkl, 'rb') as file:
                    self.ABABs[i] = pickle.load(file)
            else:
                self.ABABs[i] = []
            
            
    def __len__(self):
        return len(self.conv_dirs)

    def load_all_crops(self, i, ret_text=True):
        crops = self.crops[i]
        outs = []
        for crop in crops:
            outs.append(
                self.__getitem__(i,
                    pred_idx=crop['pred_idx'],
                    ref_styles=crop['ref_styles'],
                    ref_style=crop['ref_style'],
                    ret_text=ret_text
                )
            )
        return outs

    def load_all_ABABs(self, i, ret_text=True):
        crops = self.ABABs[i]
        outs = []
        for crop in crops:
            outs.append(
                self.__getitem__(i,
                    pred_idx=crop['pred_idx'],
                    ref_styles=crop['ref_styles'],
                    ref_style=crop['ref_style'],
                    ret_text=ret_text
                )
            )
        return outs

    def __getitem__(self, i, 
        pred_idx=None,
        ref_styles=None,
        ref_style=None,
        ret_text=False
    ):
        texts = self.texts[i]
        styles = self.styles[i]
        spk_styles = self.spk_styles[i]
        audio_paths = self.audio_paths[i]
        
        N = len(texts)
        if pred_idx != None:
            j = pred_idx
        else:
            j = random.randint(1, N-1) # end
        j0 = max(0, j-self.max_turn)
         
        hist_texts = '\n'.join(texts[j0:j])
        hist_styles = styles[j0:j]
        last_audio_path = audio_paths[j-1]
        last_spk = texts[j-1].split(':')[0]
            
        next_text = texts[j]
        next_spk = next_text.split(':')[0]
        # next_spk = next_text[0:6]
        target = next_text.split("TEXT:", 1)[1].strip() 
 
        # If a speaker only speaks once in this conversation, we cannot 
        # sample a different style from the output target as the reference, 
        # so we skip this sample.
        if len(spk_styles[next_spk.split('_')[1]]) == 1:
            return self.__getitem__(i, pred_idx=pred_idx, 
            ref_styles=ref_styles, ref_style=ref_style, 
            ret_text=ret_text)
        
        # Load the target style and sample another style as reference
        tar_style = styles[j]
        
        if ref_styles == None:
            # Sample reference styles of all speakers.
            ref_styles = []
            ref_style_prompt = ''
            for spk in sorted(spk_styles.keys()):
                s = torch.tensor(random.choice(spk_styles[spk]))
                if spk == next_spk.split('_')[1]: # Avoid sampling the target style
                    while torch.equal(s, tar_style):
                        s = torch.tensor(random.choice(spk_styles[spk]))
                    ref_style = s
                ref_style_prompt += f' spk_{str(spk)}: STYLE: <|extra_123|>'
                ref_styles.append(s)
            ref_styles = torch.stack(ref_styles, dim=0)

        else:
            assert ref_style != None # also need to provide ref_style
            # Fix ref_style 
            ref_style = hist_styles[1]
            ref_styles_ = []
            ref_style_prompt = ''
            for spk in sorted(ref_styles.keys()):
                if spk == next_spk.split('_')[1]:
                    # print(spk, ref_style, hist_styles[1])
                    s = ref_style
                else:
                    s = torch.tensor(ref_styles[spk])
                ref_style_prompt += f' spk_{str(spk)}: STYLE: <|extra_123|>'
                ref_styles_.append(s)
            ref_styles = torch.stack(ref_styles_, dim=0)
                        
        in_styles = torch.cat(
            [ref_styles, hist_styles], 0
        )

        prompt = input_template.format(
            speakers_and_styles=ref_style_prompt,
            hist=hist_texts,
            next_spk=next_spk,
        )
        prompt = prompt.replace(': STYLE: <|extra_123|>', '')
    
        if not self.tokenize:
            return dict(
                prompt=prompt,
                target=target,
            )
        
        ret = preprocess(
            [[
                {
                    "from": "user",
                    "value": prompt
                },
                {
                    "from": "assistant",
                    "value": target
                }
            ]], 
            self.tokenizer, 
            self.max_len
        )

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )

        if ret_text:
            ret['prompt'] = prompt
            ret['target'] = target
            ret['pred_idx'] = j
            ret['conv_idx'] = last_audio_path.split('/')[-2]
            ret['tar_audio_path'] = audio_paths[j]
            ret['last_audio_path'] = audio_paths[j-1]
            ret['pre_audio_path'] = audio_paths[j0:j]
            ret['ref_style'] = ref_style
            ret['hist_styles'] = hist_styles

        return ret
