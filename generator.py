import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_PROMPT = (
"You are a helpful assistant. Given a question and supporting context, answer concisely.\n" \
"If the answer is not in the context, say 'I don't know'.\n" \
"\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
)


class LLMGenerator:
    def __init__(self, model_name, max_new_tokens=128, temperature=0.01, top_p=0.95):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 4-bit if CUDA available
        quant_args = {"device_map": "auto"}
        if torch.cuda.is_available():
            quant_args.update({
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            })
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **quant_args)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    
    
    @torch.no_grad()
    def generate(self, question, context_text, prompt_template=DEFAULT_PROMPT):
        prompt = prompt_template.format(question=question, context=context_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Answer:")[-1].strip()
