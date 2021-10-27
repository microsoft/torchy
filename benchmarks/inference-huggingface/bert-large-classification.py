from testdriver import *
from transformers import pipeline, TextClassificationPipeline, BertTokenizer, BertForSequenceClassification

txt = [
  'I hate you',
  'I love you',
  'The sun looks beautiful today',
  'So call me maybe',
  'This merchant is crap',
  'Worst purchase of my life!!',
  "Don't be rude sir",
]
txt = txt * int(300 / len(txt))

#nlp = pipeline('sentiment-analysis')
nlp = TextClassificationPipeline(
  model=BertForSequenceClassification.from_pretrained('bert-large-uncased'),
  tokenizer=BertTokenizer.from_pretrained('bert-large-uncased'),
  device=0 if cuda else -1
)

for s in txt:
  result = nlp(s)[0]
  print(f'{s}, {result["label"]}, {round(result["score"], 3)}')
