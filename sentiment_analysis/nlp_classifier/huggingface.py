# For testing...

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


results = classifier(
    "POOR SERVICE!! I shopped elsewhere. Another big chain store but money may hopefully stayed more local. Thank you for reading."
)
print(results)
