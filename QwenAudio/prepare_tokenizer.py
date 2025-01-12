from transformers import AutoTokenizer

def get_tokenizer(in_style_tok='<|extra_123|>', out_style_tok='<|extra_124|>'):

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen-Audio-Chat', trust_remote_code=True
    )

    # Use a pre-allocated special token as the style placeholder
    tokenizer.add_special_tokens(
        {'additional_special_tokens': [in_style_tok, out_style_tok]}
    )

    tokenizer.pad_token_id = tokenizer.eod_id

    return tokenizer

def get_in_style_id(tokenizer, in_style_tok='<|extra_123|>'):
    return tokenizer.convert_tokens_to_ids(in_style_tok)

def get_out_style_id(tokenizer, out_style_tok='<|extra_124|>'):
    return tokenizer.convert_tokens_to_ids(out_style_tok)
