from nltk.tokenize import word_tokenize
import nltk
import pickle


file = open("word_features","rb")
file2 = open("Sentiment_analysis_movie_model","rb")
model = pickle.load(file2)
word_features = pickle.load(file)
file.close()
file.close()


def find_features(document):
    features = {}
    doc_words = word_tokenize(document)
    for w in doc_words:
        features[w] = (w in word_features)
    return features


def Sentiments(text):
    features = find_features(text)
    return(model.classify(features))

