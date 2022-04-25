import matplotlib.pyplot as plt
from pca import PCA
def generate_graph(input_wv, target_word, n_neighbors=10):
    # get the nearest 10 words to the target word
    target_word_neighbors = input_wv.most_similar(target_word, topn=n_neighbors)
    print(f"{target_word} had the following neighbors: {target_word_neighbors}")
    # get the word2vec vectors for the neighbors
    vecs = map(lambda x: input_wv[x[0]], target_word_neighbors)
    # get the word2vec vectors for the target word
    target_word_vec = input_wv[target_word]
    # add the target word to the list of vectors
    vecs = list(vecs)
    vecs.append(target_word_vec)
    # perform pca on the vectors
    pca_vecs = PCA(vecs)
    print("Result of PCA")
    print(pca_vecs)
    # plot the vectors in 2d
    plt.scatter(pca_vecs[:,0], pca_vecs[:,1])
    # label the points with the words
    for i, word in enumerate(target_word_neighbors):
        plt.annotate(word[0], (pca_vecs[i,0], pca_vecs[i,1]))
    # add the target word to the plot and color it red
    plt.scatter(pca_vecs[-1,0], pca_vecs[-1,1], color="red")
    plt.annotate(target_word, (pca_vecs[-1,0], pca_vecs[-1,1]))
    # label the bottom axis "PC1"
    plt.xlabel("PC1")
    # label the left axis "PC2"
    plt.ylabel("PC2")

    # add title to plot
    # bold the title with larger font size
    # move axis scale to the top
    plt.title(f"{target_word}'s {n_neighbors} nearest neighbors", fontweight="bold", y=1.1)
    # save the plot with 16:9 aspect ratio
    plt.savefig(f"graphs/{target_word}_neighbors.png", bbox_inches="tight", dpi=300)
    plt.show()

def generate_graphs(input_wv, target_word_list):
    # generates nn graphs using matplotlib
    # input_wv: word2vec keyedvectors
    # target_word_list: list of words to generate graphs for
    for word in target_word_list:
        generate_graph(input_wv, word)

