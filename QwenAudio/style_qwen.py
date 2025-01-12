'''
Adapted from Qwen-Audio
https://github.com/QwenLM/Qwen-Audio/blob/main/modeling_qwen.py
Author: Xilin Jiang xj2289@columbia.edu
'''

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import copy
from typing import TYPE_CHECKING, Optional, Tuple, Union, \
    Callable, List, Any, Generator, Dict

from transformers import AutoModelForCausalLM, AutoConfig, \
    PreTrainedTokenizer, GenerationConfig, GenerationMixin, \
        StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .qwen_generation_utils import (
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)
from .modeling_qwen import _ERROR_BAD_CHAT_FORMAT, \
_SENTINEL, _ERROR_STREAM_IN_CHAT


class QwenAudioWithStyles(nn.Module, GenerationMixin):
    
    def __init__(self, 
        qwen, 
        style_emb_dim=256, 
        text_emb_dim=4096,
        lambda_style=1, 
        in_style_id=151769,
        out_style_id=151770,
        style_loss_fn=nn.MSELoss(reduction='mean'),
        style_loss_dim=128
    ):
        super().__init__()
        
        self.qwen = qwen
        self.config = qwen.config
        self.main_input_name = qwen.main_input_name
        self.device = qwen.device
        self.generation_config = qwen.generation_config
        # NEW!!!
        self.style_in_project = nn.Linear(style_emb_dim, text_emb_dim)
        self.style_out_project = nn.Linear(text_emb_dim, style_emb_dim)
        self.style_loss_fn = style_loss_fn
        self.lambda_style = lambda_style

        self.est_style = None
        self.in_style_id = in_style_id
        self.out_style_id = out_style_id

        self.style_loss_dim = style_loss_dim
         
    def can_generate(self):
        return True

    def transformer_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_info: Dict = None,
        in_styles = None, # NEW!!!
    ):

        assert in_styles != None
        if past_key_values is None and torch.any(input_ids == self.qwen.transformer.config.audio['audio_start_id']):
            bos_pos = torch.where(input_ids == self.qwen.transformer.config.audio['audio_start_id'])
            eos_pos = torch.where(input_ids == self.qwen.transformer.config.audio['audio_start_id'] + 1)
            assert (bos_pos[0] == eos_pos[0]).all()
            audio_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
            if isinstance(audio_info, Dict):
                audios = audio_info["input_audios"]
                audio_span_tokens = audio_info["audio_span_tokens"]
                input_audio_lengths = audio_info["input_audio_lengths"]
                audios = self.qwen.transformer.audio.encode(audios,input_audio_lengths, audio_span_tokens)
            else:
                audios = torch.concat([_["input_audios"] for _ in audio_info])
                input_audio_lengths = torch.concat([_["input_audio_lengths"] for _ in audio_info])
                audio_span_tokens = []
                for _ in audio_info:
                    audio_span_tokens.extend(_['audio_span_tokens'])
                audios = self.qwen.transformer.audio.encode(audios, input_audio_lengths, audio_span_tokens)

        else:
            audios = None

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.qwen.transformer.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.qwen.transformer.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.qwen.transformer.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.qwen.transformer.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.qwen.transformer.h))
        else:
            if self.qwen.transformer.use_cache_quantization:
                past_length = past_key_values[0][0][0].size(2)
            else:
                past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.qwen.transformer.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.qwen.transformer.dtype).min

        encoder_attention_mask = None
        head_mask = self.qwen.transformer.get_head_mask(head_mask, self.qwen.transformer.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.qwen.transformer.wte(input_ids)
        
        ################################################################################

        # Project style embeddings
        in_style_embeds = [self.style_in_project(style) 
                        for style in in_styles]
        
        # Find the location of input style placeholders
        in_style_locs = [
            (input_ids[b]==self.in_style_id).nonzero() for b in range(len(in_styles))
        ]

        # Replace the token embeddings of '<|extra_123|>'
        # by projected style embeddings of the same dimension
        b = 0 # Iterate over batch_size
        for in_style_loc, in_style_embed in zip(in_style_locs, in_style_embeds):
            if in_style_loc.numel():
                inputs_embeds[b, in_style_loc.squeeze(1)] = in_style_embed
            b += 1
 
        ################################################################################
            
        hidden_states = inputs_embeds
        
        kv_seq_len = hidden_states.size()[1]
        if past_key_values[0] is not None:
            # past key values[0][0] shape: bs * seq_len * head_num * dim
            if self.qwen.transformer.use_cache_quantization:
                kv_seq_len += past_key_values[0][0][0].shape[2]
            else:
                kv_seq_len += past_key_values[0][0].shape[1]

        if self.qwen.transformer.training or not self.qwen.transformer.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        elif kv_seq_len != hidden_states.size()[1]:
            ntk_alpha_list = self.qwen.transformer.rotary_emb._ntk_alpha_cached_list
        else:
            ntk_alpha_list = []
            if attention_mask is not None and kv_seq_len > self.qwen.transformer.seq_length:
                true_seq_lens = attention_mask.squeeze(1).squeeze(1).eq(0).sum(dim=-1, dtype=torch.int32)
                for i in range(hidden_states.size()[0]):
                    true_seq_len = true_seq_lens[i].item()
                    ntk_alpha = self.qwen.transformer.get_ntk_alpha(true_seq_len)
                    ntk_alpha_list.append(ntk_alpha)
            else:
                ntk_alpha = self.qwen.transformer.get_ntk_alpha(kv_seq_len)
                ntk_alpha_list.append(ntk_alpha)
        self.qwen.transformer.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.qwen.transformer.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
        ]

        hidden_states = self.qwen.transformer.drop(hidden_states)
        if audios is not None:
            for idx, (i, a, b) in enumerate(audio_pos):
                hidden_states[i][a : b+1] = audios[idx]
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.qwen.transformer.gradient_checkpointing and self.qwen.transformer.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.qwen.transformer.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.qwen.transformer.gradient_checkpointing and self.qwen.transformer.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    rotary_pos_emb_list,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    rotary_pos_emb_list=rotary_pos_emb_list,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        hidden_states = self.qwen.transformer.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_info: Dict = None,
        in_styles = None, # NEW!!!
        tar_style = None, # NEW!!!
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        assert in_styles != None

        return_dict = (
            return_dict if return_dict is not None else self.qwen.config.use_return_dict
        )

        # print(input_ids.shape)

        transformer_outputs = self.transformer_forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            audio_info=audio_info,
            in_styles=in_styles
        )
        hidden_states = transformer_outputs[0] # (B, L, D)
        lm_logits = self.qwen.lm_head(hidden_states)
        
        ################################################################################

        # Find where to predict output style
        out_style_loc = [
            (input_ids[b]==self.out_style_id).nonzero().squeeze() for b in range(len(in_styles))
        ]

        if out_style_loc[0].numel():
            # Extract the hidden state at <|extra_124|>. Shape (B, D).
            style_hidden_state = hidden_states[torch.arange(hidden_states.shape[0]), out_style_loc]

            # Project to the same dim as the target style
            est_style = self.style_out_project(style_hidden_state)
            self.est_style = est_style # hold the result here for later access 

            if tar_style != None:
                style_loss = self.lambda_style * self.style_loss_fn(
                    est_style[:, -self.style_loss_dim:], 
                    tar_style[:, -self.style_loss_dim:]
                )
            else:
                style_loss = torch.tensor(0).to(est_style.device)
        else:
            style_loss = torch.tensor(0).to(lm_logits.device)

        ################################################################################
        
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
    
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if tar_style != None:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            ), style_loss
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        audio_info = kwargs.pop("audio_info", None)
        token_type_ids = kwargs.get("token_type_ids", None)
        in_styles = kwargs.get("in_styles", None)

        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "audio_info": audio_info,
                "in_styles": in_styles,
                "tar_style": None
            }
        )

        return model_inputs

    def run(
        self,
        inputs: Dict,
        tokenizer: PreTrainedTokenizer,
        system: str = "You are a helpful assistant.",
        stream: Optional[bool] = _SENTINEL,
        stop_words_ids: Optional[List[List[int]]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        generation_config = generation_config if generation_config is not None \
            else self.qwen.generation_config

        assert stream is _SENTINEL, _ERROR_STREAM_IN_CHAT
        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT

        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
 
        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))

        raw_text, context_tokens, audio_info = make_context(
            tokenizer,
            inputs['prompt'],
            history=[],
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        input_ids = torch.tensor([context_tokens]).to(self.device)
        in_styles = [inputs['in_styles'].to(self.device)]

        kwargs['audio_info'] = audio_info
        kwargs['in_styles'] = in_styles

        outputs = self.generate(
            inputs=input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            **kwargs,
        )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace',
            audio_info=audio_info
        )

        return response, self.est_style

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config = generation_config if generation_config is not None \
            else self.qwen.generation_config

        # Process stop_words_ids.
        stop_words_ids = kwargs.pop("stop_words_ids", None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)



        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )
