from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
stemmer=PorterStemmer()
import tensorflow as tf
nltk.download('punkt')  
def tokenize(text_data):

    

# Tokenize the text data
    tokenized_data = [word_tokenize(text_data)]


    return tokenized_data
def stemm(text_data):
    return [stemmer.stem(word) for word in text_data]
def bow(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stemmer.stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag