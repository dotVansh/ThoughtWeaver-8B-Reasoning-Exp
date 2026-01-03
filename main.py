import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer

model_id = "vazirani/ThoughtWeaver-8B-Reasoning-Exp"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map=device)


messages = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "In your opinion, what differentiates Epicureanism and hedonism, and where on that spectrum does most of humanity lie?",
    }]
}]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
tokenized.pop("token_type_ids", None)

for key in tokenized:
    if isinstance(tokenized[key], torch.Tensor):
        tokenized[key] = tokenized[key].to(device)

output = model.generate(
    **tokenized,
    max_new_tokens=8192,
)[0]
decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)