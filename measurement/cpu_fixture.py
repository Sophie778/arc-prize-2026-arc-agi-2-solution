"""Create a 3,392 parameter real Qwen3 for CPU control checks (MIT-0)."""
def build_tiny(model):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    model.mkdir(parents=True,exist_ok=True)
    vocab={'[UNK]':0,'[PAD]':1,'</think>':2,'[EOS]':3}
    vocab.update({f'word{i}':i for i in range(4,32)})
    tokenizer=Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]',
        eos_token='[EOS]',additional_special_tokens=['</think>'],model_max_length=32768)
    tok.save_pretrained(model);torch.manual_seed(0)
    net=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=32768,
        eos_token_id=3,pad_token_id=1))
    net.save_pretrained(model)
    return model
