import numpy as np
from gensim.models import KeyedVectors
from model.data_processing import *

# To run this file, you need the pre-trained Word2Vec model downloaded from
# http://vectors.nlpl.eu/repository/
# the model with ID 40 (English CoNLL17 corpus with vocab size of 4027169)
# Find the .bin file and rename it to 'model.bin'

EMBEDDING_FILE = 'model.bin'
print('Indexing word vectors')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


def embedding_matrix_creator(embedding_dimension, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
    for word, index in word_index.items():
        try:
            embedding_vector = word2vec[word]
            embedding_matrix[index] = embedding_vector
        except KeyError:
            pass
    # words not found in embedding index will be all-zeros.
    return embedding_matrix


embedding_matrix = embedding_matrix_creator(100,
                                            word_index=tokenizer.word_index)
embedding_matrix.dump("embedding_matrix.dat")
