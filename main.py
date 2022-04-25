from graph import generate_graphs
from preprocessing import brown_corpus, load_w2v_vectors, prep
import os
if __name__ == '__main__':
    vectors_path = brown_corpus(embed=True)
    # load the word2vec vectors
    word2vec_vectors = load_w2v_vectors(vectors_path)
    # generate graphs
    target_words = ["linear", "university", "work", "graduation", "college", "final", "project"]
    generate_graphs(word2vec_vectors, target_words)