from testdriver import *
from transformers import pipeline

txt = [
  'I hate you',
  'I love you',
  'The sun looks beautiful today',
  'So call me maybe',
  'This merchant is crap',
  'Worst purchase of my life!!',
  "Don't be rude sir",
]
txt = txt * int(1000 / len(txt))

nlp = pipeline('sentiment-analysis')

for s in txt:
  result = nlp(s)[0]
  print(f'{s}, {result["label"]}, {round(result["score"], 3)}')
