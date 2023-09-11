from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model_n_tokenizer(model_name, trust_remote_code=True, low_cpu_mem_usage=False):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    except Exception as e:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer