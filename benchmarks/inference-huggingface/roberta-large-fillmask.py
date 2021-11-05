from testdriver import *
from transformers import pipeline, FillMaskPipeline, RobertaForMaskedLM, RobertaTokenizer

if torchscript:
  print('TorchScript not supported!')
  exit(-1)

txt = [
  "Hello I'm a <mask> model.",
  "Writing dummy sentences for completion is a lot of <mask>.",
  "The weather in <mask> looks terrible today.",
  "We love spending our hollidays in <mask>.",
  "The best NLP model is <mask>.",
  "PyTorch is an <mask> machine learning library.",
]
txt = txt * int(1000 / len(txt))

#unmasker =  pipeline('fill-mask', model='roberta-large')
unmasker = FillMaskPipeline(
  model=RobertaForMaskedLM.from_pretrained('roberta-large'),
  tokenizer=RobertaTokenizer.from_pretrained('roberta-large'),
  device=0 if cuda else -1
)

for s in txt:
  results = unmasker(s, top_k=2)
  print(f'{s} <- {results[0]["token_str"]} / {results[1]["token_str"]}')
