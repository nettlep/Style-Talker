'''
Adapted from Qwen-Audio Finetuning
https://github.com/QwenLM/Qwen-Audio/issues/38
Author: Xilin Jiang xj2289@columbia.edu
'''

import os
import sys
import wandb
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from QwenAudio.style_qwen import QwenAudioWithStyles
from QwenAudio.trainer import TrainerWithStyles

if __name__ == '__main__':
   
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    os.environ['WANDB_PROJECT'] = hparams['project']
    hparams['lora_qwen'].print_trainable_parameters()

    style_qwen = QwenAudioWithStyles(
        qwen=hparams['lora_qwen'],
        lambda_style=hparams['lambda_style'],
        in_style_id=hparams['in_style_id'],
        out_style_id=hparams['out_style_id'],
        style_loss_fn=torch.nn.L1Loss(),
        style_loss_dim=hparams['style_loss_dim']
    )
    
    trainer = TrainerWithStyles(
        model=style_qwen,
        tokenizer=hparams['tokenizer'],
        train_dataset=hparams['train_data'],
        eval_dataset=hparams['valid_data'],
        data_collator=hparams['data_collator'],
        args=hparams['train_config'],
    )
    
    try:
        trainer.train()
        trainer.save_state()
        wandb.finish()
    except Exception as E:
        print(E)
        wandb.finish()
