from logging import debug

from two_classifier import BertClassifier

positive_score, neutral_score, negative_score = BertClassifier("hate").return_list()
print(positive_score, neutral_score, negative_score)
