import os

import torch
from torch import nn

from dataclasses import dataclass
from typing import Dict, Tuple, Union, Optional, List

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model

from QwenAudio.style_qwen import QwenAudioWithStyles
from QwenAudio.trainer import load_lora


def move_dict_to_device(inputs, device):
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    elif isinstance(inputs, dict):
        return {key: move_dict_to_device(value, device) for key, value in inputs.items()}
    else:
        return inputs


class StyleQwen(nn.Module):

    def __init__(self,
        ckpt_root,
        model_name='Qwen/Qwen-Audio-Chat',
        bf16=True,
        lora_r=16,
        lora_modules=['c_attn', 'attn.c_proj', 'w1', 'w2', 'query', 'key', 'value'],
        max_window_size=1536,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        # Load config
        config = transformers.AutoConfig.from_pretrained(
            model_name, trust_remote_code=True,
        )
        config.use_cache = False
        
        # Load model
        qwen = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device,
            bf16=bf16, trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Use a pre-allocated special token as the style placeholder
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['<|extra_123|>', '<|extra_124|>']}
        )
        in_style_id = tokenizer.convert_tokens_to_ids('<|extra_123|>')
        out_style_id = tokenizer.convert_tokens_to_ids('<|extra_124|>')
        print(f'in_style_id: {str(in_style_id)}, out_style_id: {str(out_style_id)}')
        qwen.resize_token_embeddings(len(tokenizer)) # no change
        tokenizer.pad_token_id = tokenizer.eod_id

        # Add LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=16,
            target_modules=lora_modules,
            lora_dropout=0.05,
            bias='none'  
        )

        lora_qwen = get_peft_model(qwen, lora_config)
        lora_qwen.print_trainable_parameters()
        
        style_qwen = QwenAudioWithStyles(
            qwen=lora_qwen, 
            in_style_id=in_style_id,
            out_style_id=out_style_id
        )
        
        if ckpt_root == None:
            print('Initialized pretrained Qwen-Audio with random style projections.')
        else:
            # Load finetuned weights
            load_lora(style_qwen.qwen, os.path.join(ckpt_root, 'lora.pt'))
            style_qwen.style_in_project.load_state_dict(
                torch.load(
                    os.path.join(ckpt_root, 'style_in_project.pt'),
                    map_location=torch.device(device)
                )
            )
            style_qwen.style_out_project.load_state_dict(
                torch.load(
                    os.path.join(ckpt_root, 'style_out_project.pt'),
                    map_location=torch.device(device)
                )   
            )
        
        self.style_qwen = style_qwen.to(device)
        self.tokenizer = tokenizer
        self.max_window_size = max_window_size
        self.device = device


    def forward(self, x):
        x = move_dict_to_device(x, self.device)
        return self.style_qwen.run(
            inputs=x, 
            tokenizer=self.tokenizer, 
            max_window_size=self.max_window_size
        )
