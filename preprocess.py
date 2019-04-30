from gensim.parsing.preprocessing import STOPWORDS
from nltk import PorterStemmer, WordNetLemmatizer
from spacy.lemmatizer import Lemmatizer
from textblob import TextBlob
import pandas as pd
import re

def cleantext(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text 

def preprocessdata(essay):
    essay_processed = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    for k,v in essay.items():
        clean = cleantext(str(v))
        doc = TextBlob(str(clean))
        essay_processed.append([stemmer.stem(str(lemmatizer.lemmatize(w))) for w in doc.words if w not in STOPWORDS and w.isalpha() and len(w)>2])
    essay_processed = [" ".join(w) for w in essay_processed]
    return essay_processed

if __name__ == "__main__":
        #preprocessdata
        train_essay = pd.read_csv("/mnt/1f2870f0-1578-4534-b33f-0817be64aade/projects/Hackerearth/incedo_nlpcadad7d/incedo_participant/train_dataset.csv")
        essay_Set = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
        data_dict = {}
        c = 0
        for i in essay_Set:
            c = c+1
            set_filter = (train_essay.Essayset == 5.0)
            preprocessdata(train_essay[set_filter]['EssayText'])
            break
