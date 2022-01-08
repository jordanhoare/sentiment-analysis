import csv
import urllib.request

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task = "sentiment"
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels = []
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode("utf-8").split("\n")
    csvreader = csv.reader(html, delimiter="\t")
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
save_dir = "sentiment_analysis/nlp_classifier"
model = AutoModelForSequenceClassification.from_pretrained(save_dir)
model.save_pretrained(save_dir)

text = "i love you 😊"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
print(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
positive_label = labels[ranking[0]]
neutral_label = labels[ranking[1]]
negative_label = labels[ranking[2]]
positive_score = scores[ranking[0]]
neutral_score = scores[ranking[1]]
negative_score = scores[ranking[2]]
print(negative_score)

# for i in range(scores.shape[0]):
#     l = labels[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")
