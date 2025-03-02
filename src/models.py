from transformers import BertTokenizer, BertForSequenceClassification
import torch

class DataModel():
    """
    Class to retrieve, treat and customize the data
    """
    def __init__(self):
        pass

class BERT():
    """
    Hugging Face BERT: Bidirectional Encoder Representations from Transformers
    https://huggingface.co/docs/transformers/model_doc/bert

    For fine-tune hyperparameters:
    https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
    https://huggingface.co/datasets/gxb912/large-twitter-tweets-sentiment
    https://huggingface.co/datasets/cardiffnlp/tweet_eval
    https://huggingface.co/datasets/mteb/tweet_sentiment_extraction
    """
    def __init__(self, modelName: str):
        self.tokenizer = BertTokenizer.from_pretrained(modelName)
        self.model = BertForSequenceClassification.from_pretrained(modelName, output_attentions= True)
    
    def getSentiment(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors = 'pt', truncation = True, padding = True)
        with torch.no_grad():
            # torch - PyTorch context manager
            # PyTorch tracks gradients for backpropagation, necessary for training.
            # no_grad - Not to store gradients during the prediction (saves memory and speeds up computation)
            outputs = self.model(**inputs)

        # outputs.logits - raw outputs from BERT, not yet probabilities
        # softmax - activation layer. Converts logits into probabilities. They sum up 1.
        scores = outputs.logits.softmax(dim=1).tolist()[0]

        labels = ["very negative", "negative", "neutral", "positive", "very positive"]
        sentiment = labels[scores.index(max(scores))]

        return {"sentiment": sentiment, "scores": dict(zip(labels, scores))}
    
if __name__ == "__main__":
    bertModel = BERT("nlptown/bert-base-multilingual-uncased-sentiment")
    print(bertModel.getSentiment("I am not sure"))