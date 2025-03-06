'''
Adapted from Huggingface Transformer Trainer
Author: Xilin Jiang xj2289@columbia.edu
'''

import os
import torch
import wandb
from transformers import Trainer
from peft import set_peft_model_state_dict, get_peft_model_state_dict

from typing import Dict, Tuple, Union, Optional, List

def save_lora(lora_model, path):
    peft_state_dict = get_peft_model_state_dict(lora_model)
    torch.save(peft_state_dict, path)

def load_lora(lora_model, path):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    peft_state_dict = torch.load(path, map_location=torch.device(device))
    result = set_peft_model_state_dict(lora_model, peft_state_dict)
    print(f'LoRA Loaded from {path}.')

    
class TrainerWithStyles(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs, style_loss = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        ################################################################################
            
        if 'wandb' in self.args.report_to:
            wandb.log(
                data={"train/CE": float(loss), "train/Style": float(style_loss)},
            )

        loss = loss + style_loss
        
        ################################################################################
            
        return (loss, outputs) if return_outputs else loss

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        ################################################################################
        
        save_lora(
            self.model.qwen, 
            os.path.join(output_dir, 'lora.pt')
        )
        torch.save(
            self.model.style_in_project.state_dict(), 
            os.path.join(output_dir, 'style_in_project.pt')
        )
        torch.save(
            self.model.style_out_project.state_dict(), 
            os.path.join(output_dir, 'style_out_project.pt')
        )
        
        ################################################################################
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)   


class TrainerLoRA(Trainer):
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        ################################################################################
            
        if 'wandb' in self.args.report_to:
            wandb.log(
                data={"train/CE": float(loss), "train/Style": float(style_loss)},
            )
        
        ################################################################################
            
        return (loss, outputs) if return_outputs else loss

    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")

        ################################################################################
        
        save_lora(
            self.model, 
            os.path.join(output_dir, 'lora.pt')
        )
        
        ################################################################################
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)   
