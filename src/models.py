from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from textblob import TextBlob
from dataclasses import dataclass

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

    Self-attention Mechanism:
        - relationships between words
        - process all words at one (parallel processing)
    Multi-head attention: grammar and meaning.
    Positional Encoding: order in the sentence matters.

    Insights:
        1. Tokenizes text.
        2. Passes it through multiple transformer layers.
        3. Understands relationships between words.
        4. Outputs "logits" for different sentiment classes.
        5. Final softmax layer to convert logits into probabilities.
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
    
class VADER(): # Valence Aware Dictionary and sEntiment Reasoner
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def getSentiment(self, text: str):
        scores = self.sia.polarity_scores(text)
        return scores # (neg, pos, compound): see compound if > 0.05 positive sentiment, < -0.05 negative sentiment, between -0.05 and 0.05 neutral. Goes from -1 to 1

class TEXTBLOB():
    """
    TextBlob provides a simple way to analyze the sentiment of text. It returns two main values:
    - Polarity: A value between -1.0 (negative) and 1.0 (positive).
    - Subjectivity: A value between 0.0 (objective - fact-based) and 1.0 (subjective - opinion-based).

    It analizes the text sentence by sentence.
    iterate over blob.sentences
    """
    def getSentiment(self, text: str):
        blob = TextBlob(text)
        scores = blob.sentiment
        return {"polarity": scores.polarity, "subjectivity": scores.subjectivity}

class DEEPMOJI():
    pass

@dataclass
class Sentiment():
    opinion: str
    biased: bool
    emotion: str
    
if __name__ == "__main__":
    bertModel = BERT("nlptown/bert-base-multilingual-uncased-sentiment")
    vaderModel = VADER()
    textblobModel = TEXTBLOB()

    text = "I am not sure"

    print(f"Bert Model: {bertModel.getSentiment(text)}")
    print(f"Vader Model: {vaderModel.getSentiment(text)}")
    print(f"TextBlob Model: {textblobModel.getSentiment(text)}")