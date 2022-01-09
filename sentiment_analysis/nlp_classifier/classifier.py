import csv
import urllib.request

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

# Preprocess phrase (username and link placeholders)
def preprocess(phrase):
    new_phrase = []

    for t in phrase.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_phrase.append(t)
    return " ".join(new_phrase)


# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary


# PT
save_dir = "sentiment_analysis/nlp_classifier"
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
labels = ["negative", "neutral", "positive"]
model = AutoModelForSequenceClassification.from_pretrained(save_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained(save_dir)

phrase = "I like you. I love you"
phrase = preprocess(phrase)
encoded_input = tokenizer(phrase, return_tensors="pt")
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
# ranking = ranking[::-1]
# negative_label = labels[2]
# neutral_label = labels[1]

positive_score = "{:.2%}".format(scores[2])
neutral_score = "{:.2%}".format(scores[1])
negative_score = "{:.2%}".format(scores[0])

print(phrase)
print(positive_score)
print(neutral_score)
print(negative_score)
