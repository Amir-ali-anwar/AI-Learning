from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
def load_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Loads a DistilBERT model and tokenizer for sequence classification.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

