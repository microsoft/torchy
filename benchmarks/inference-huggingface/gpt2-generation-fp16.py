from testdriver import *
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, GPT2Tokenizer

txt = [
  "Hello, I'm a language model,",
  "I'm Portuguese, therefore I love",
  "PHP, the best programming language for the web, is",
  "The lottery numbers for the next Friday are:",
  "PyTorch provides two high-level features:",
  "A compiler is a program that",
  "The weather today looks",
]
txt = txt * int(200 / len(txt))

#generator = pipeline('text-generation', model='gpt2')
generator = TextGenerationPipeline(
  model=GPT2LMHeadModel.from_pretrained('gpt2', torch_dtype=torch.float16),
  tokenizer=GPT2Tokenizer.from_pretrained('gpt2'),
  device=0 if cuda else -1
)

for s in txt:
  r = generator(s, max_length=30, num_return_sequences=1, pad_token_id=0)
  print(r[0]['generated_text'])
