# https://huggingface.co/bert-base-cased
# https://huggingface.co/transformers/v2.4.0/quickstart.html
# https://chriskhanhtran.github.io/posts/bert-for-sentiment-analysis/
# https://github.com/curiousily/Deploy-BERT-for-Sentiment-Analysis-with-FastAPI/blob/master/sentiment_analyzer/classifier/sentiment_classifier.py


import json

import torch
import torch.nn.functional as tf
from torch import nn
from transformers import BertModel, BertTokenizer

# Open model config
with open("sentiment_analysis/nlp_classifier/config.json") as json_file:
    config = json.load(json_file)


class BertClassifier:
    """
    (1) Tokenization and Input Formatting
    (2) Create PyTorch DataLoader
    (3) Create BertClassifier
    (4) Set learning rate & initialise classifier
    (5) Training loop
    (6) Evaluation
    """

    def __init__(self, n_classes):
        """
        Instantiate the tokenizer
        """

        super(BertClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True,
        )
        self.bert = BertModel.from_pretrained(config["BERT_MODEL"])
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def preprocessing(self, phrase):
        """
        In order to apply the pre-trained BERT, we must use the tokenizer
        provided by the library. This is because (1) the model has a
        specific, fixed vocabulary and (2) the BERT tokenizer has a
        particular way of handling out-of-vocabulary words.
        """
        # Create empty lists to store outputs
        self.input_ids = []
        self.attention_masks = []

        self.encoded_phrase = self.tokenizer.encode_plus(
            phrase,  # Preprocess sentence
            add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
            pad_to_max_length=True,  # Pad sentence to max length
            return_tensors="pt",  # Return PyTorch tensor
            return_attention_mask=True,  # Return attention mask
            max_length=config["MAX_SEQUENCE_LEN"],
        )

        # Add the outputs to the lists
        self.input_ids = self.encoded_phrase["self.input_ids"].to(self.device)
        self.attention_mask = self.encoded_phrase["self.attention_mask"].to(self.device)

        self.input_ids = torch.tensor(self.input_ids)
        self.attention_masks = torch.tensor(self.attention_masks)

    # def predict(self, phrase):
    #     """
    #     Performs prediction
    #     """
    #     with torch.no_grad():
    #         probabilities = tf.softmax(
    #             self.classifier(self.input_ids, self.attention_mask), dim=1
    #         )
    #     confidence, predicted_class = torch.max(probabilities, dim=1)
    #     predicted_class = predicted_class.cpu().item()
    #     probabilities = probabilities.flatten().cpu().numpy().tolist()
    #     return (
    #         config["CLASS_NAMES"][predicted_class],
    #         confidence,
    #         dict(zip(config["CLASS_NAMES"], probabilities)),
    #     )


def get_classifier():
    return BertClassifier()
