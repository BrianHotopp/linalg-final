import numpy as np
import gensim
import re
import string
import sys
import time
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
def get_word2vec_vectors_huge(dataset):
    # stream the sentences from the dataset
    with open(dataset, 'r') as f:
        data = f.read()

# get the word2vec vectors for the dataset
def get_word2vec_vectors(dataset):
    # returns 
    # load the dataset
    with open(dataset, 'r') as f:
        data = f.read()
    # split the dataset into sentences
    sentences = nltk.sent_tokenize(data)
    # tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    # create a word2vec model
    model = gensim.models.Word2Vec(tokenized_sentences, min_count=1)
    # get the word2vec vectors for the dataset
    word2vec_vectors = model.wv
    return word2vec_vectors

def save_w2v_vectors(word2vec_vectors, filename):
    # save the word2vec vectors
    word2vec_vectors.save(filename)

def load_w2v_vectors(filename):
    # load the word2vec vectors
    word2vec_vectors = KeyedVectors.load_word2vec_format(filename, binary=False)
    return word2vec_vectors
def main():
    dataset = "coca_all.txt"
    # get the word2vec vectors for the dataset
    word2vec_vectors = get_word2vec_vectors(dataset)
    # save the word2vec vectors
    save_w2v_vectors(word2vec_vectors, "coca_all_w2v.txt")
    # print the top 10 words most similar to word
    word = "poop"
    print(word2vec_vectors.most_similar(word))
if __name__ == "__main__":
    main()