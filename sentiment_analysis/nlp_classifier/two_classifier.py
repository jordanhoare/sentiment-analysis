# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment


from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BertClassifier:
    """
    (1) Instantiate the model & tokenizer
    (2) Preprocessing/encoding
    (3) Format scores
    (4) Return list of scores
    """

    def __init__(self, input_phrase):
        save_dir = "sentiment_analysis/nlp_classifier"
        model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.save_pretrained(save_dir)
        self.preprocessing(input_phrase)
        self.formatting()

    # stringify
    #   File "d:\CompSci\Projects\sentiment-analysis\sentiment_analysis\nlp_classifier\two_classifier.py", line 27, in preprocessing
    #     for t in input_phrase.split(" "):
    # AttributeError: 'Phrases' object has no attribute 'split'

    def preprocessing(self, input_phrase):
        encoded_phrase = []
        for t in input_phrase.split(" "):
            # t = "@user" if t.startswith("@") and len(t) > 1 else t
            # t = "http" if t.startswith("http") else t
            encoded_phrase.append(t)
            encoded_phrase = " ".join(encoded_phrase)
        encoded_input = self.tokenizer(encoded_phrase, return_tensors="pt")
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        self.scores = softmax(scores)

    def formatting(self):
        self.positive_score = "{:.2%}".format(self.scores[0])
        self.neutral_score = "{:.2%}".format(self.scores[1])
        self.negative_score = "{:.2%}".format(self.scores[2])

    def return_list(self):
        return [self.positive_score, self.neutral_score, self.negative_score]
