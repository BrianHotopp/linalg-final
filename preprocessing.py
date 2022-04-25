import numpy as np
import os
import gensim
from nltk.corpus import brown
import re
import string
import sys
import time
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
# clean the dataset without loading the whole dataset
# precompile regular expressions
# remove words between 1 and 3 characters in length
r1 = re.compile(r'\b\w{1,3}\b')
# remove punctuation
r2 = re.compile(r'[^\w\s]')
# remove numbers
r3 = re.compile(r'\d+')

def clean_dataset(in_dataset, out_dataset):
    with open(in_dataset, "r") as f:
        with open(out_dataset, "a") as out:
            for line in f:
                line = line.lower()
                # remove words between 1 and 3 characters in length
                line = r1.sub('', line)
                # remove punctuation
                line = r2.sub('', line)
                # remove numbers
                line = r3.sub('', line)
                # append the line to the output file 
                out.write(line)

# get the word2vec vectors for the dataset
def get_word2vec_vectors(dataset):
    model = gensim.models.Word2Vec(corpus_file=dataset, min_count=1)
    word2vec_vectors = model.wv
    return word2vec_vectors

def save_w2v_vectors(word2vec_vectors, filename):
    # save the word2vec vectors
    word2vec_vectors.save(filename)

def load_w2v_vectors(filename):
    # load the word2vec vectors
    word2vec_vectors = KeyedVectors.load(filename)
    return word2vec_vectors

def prep(filename, clean=False):
    if clean:
        dataset = filename
        out_dataset = filename.split(".")[0] + "_clean.txt"
        # delete the output file if it exists
        try:
            os.remove(out_dataset)
        except OSError:
            pass
        # clean dataset of punctuation and numbers
        clean_dataset(dataset, out_dataset)
    else:
        out_dataset = filename
    word2vec_vectors = get_word2vec_vectors(out_dataset)
    # save the word2vec vectors
    # vectors_file_name
    v = filename.split(".")[0] + "_w2v_vectors.txt"
    save_w2v_vectors(word2vec_vectors, v)
    return v
def brown_corpus(embed=True):
    v = "brown_w2v_vectors.txt"
    if embed:
        # get the word2vec vectors for the dataset
        word2vec_vectors = Word2Vec(brown.sents()).wv
        # save the word2vec vectors
        # vectors_file_name
        save_w2v_vectors(word2vec_vectors, v)
    return v
def main():
    small = False
    clean = True
    if small:
        dataset = "coca_all_sm.txt"
        out_dataset = "coca_all_sm_clean.txt"
    else:
        dataset = "coca_all.txt"
        out_dataset = "coca_all_clean.txt"
    # delete the output file if it exists
    try:
        os.remove(out_dataset)
    except OSError:
        pass
    if clean:
        # clean dataset of punctuation and numbers
        clean_dataset(dataset, out_dataset)
    word2vec_vectors = get_word2vec_vectors(out_dataset)
    # save the word2vec vectors
    # vectors_file_name
    v = dataset.split(".")[0] + "_w2v_vectors.txt"
    save_w2v_vectors(word2vec_vectors, v)
    # print the top 10 words most similar to word
    word = "test"
    print(word2vec_vectors.most_similar(word))
if __name__ == "__main__":
    main()