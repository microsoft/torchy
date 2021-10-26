from testdriver import *
from transformers import pipeline, QuestionAnsweringPipeline, BertTokenizer, BertForQuestionAnswering

# https://en.wikipedia.org/wiki/PyTorch
context = """
PyTorch is an open source machine learning library based on the Torch library,
used for applications such as computer vision and natural language processing,
primarily developed by Facebook's AI Research lab (FAIR).
It is free and open-source software released under the Modified BSD license.
Although the Python interface is more polished and the primary focus of
development, PyTorch also has a C++ interface.

A number of pieces of deep learning software are built on top of PyTorch,
including Tesla Autopilot, Uber's Pyro, HuggingFace's Transformers,
PyTorch Lightning, and Catalyst.

PyTorch provides two high-level features:
Tensor computing (like NumPy) with strong acceleration via graphics
processing units (GPU)
Deep neural networks built on a type-based automatic differentiation system
"""

questions = [
  "What's PyTorch?",
  "Who's the primary developer of PyTorch?",
  'What does FAIR stand for?',
  'How many high-level features does PyTorch provide?',
  'Which software is built on top of PyTorch?',
  "What's the license of PyTorch?",
  'Which interfaces does PyTorch provide?',
  'What are the main applications PyTorch is used for?',
]
questions = questions * int(1000 / len(questions))

#nlp = pipeline("question-answering")
nlp = QuestionAnsweringPipeline(
  model=BertForQuestionAnswering.from_pretrained('bert-large-uncased'),
  tokenizer=BertTokenizer.from_pretrained('bert-large-uncased'),
  device=0 if cuda else -1
)

for q in questions:
  result = nlp(question=q, context=context)
  print(f'{q}, {result["answer"]}, {round(result["score"], 3)}, start: {result["start"]}, end: {result["end"]}')
