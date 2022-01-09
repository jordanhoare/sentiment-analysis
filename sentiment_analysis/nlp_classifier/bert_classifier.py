# https://huggingface.co/bert-base-cased
# https://huggingface.co/transformers/v2.4.0/quickstart.html
# https://chriskhanhtran.github.io/posts/bert-for-sentiment-analysis/
# https://github.com/curiousily/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI/blob/master/sentiment_analyzer/classifier/sentiment_classifier.py

import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BertClassifier(str):
    """
    (1) Instantiate the model & tokenizer
    (2) Preprocess phrase (username and link placeholders)
    (3) Evaluation
    """

    def Preprocessing(self, name):
        """
        Preprocess phrase (username and link placeholders)
        """
        encoded_phrase = []
        for t in name.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            encoded_phrase.append(t)
            encoded_phrase = " ".join(encoded_phrase)
        return encoded_phrase

    def __init__(self, name):
        print(1)
        save_dir = "sentiment_analysis/nlp_classifier"
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        labels = ["negative", "neutral", "positive"]
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        encoded_phrase = self.Preprocessing(name)

        encoded_input = tokenizer(encoded_phrase, return_tensors="pt")
        output = model(**encoded_input)
        self.scores = output[0][0].detach().numpy()
        self.scores = softmax(self.scores)
        positive_score = "{:.2%}".format(self.scores[0])
        neutral_score = "{:.2%}".format(self.scores[1])
        negative_score = "{:.2%}".format(self.scores[2])

    def __call__(self):
        print("fasdaf")
        return self.scores
