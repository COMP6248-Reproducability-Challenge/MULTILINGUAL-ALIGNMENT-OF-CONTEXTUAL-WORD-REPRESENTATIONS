import numpy as np


# CSLS
def get_matrix_cos_sim(e1, e2):
    """
    Get the cosine similarity of 2 vector embeddings, or lists of vector embeddings

    :param e1: embeddings vector 1
    :param e2: embeddings vector 2
    :return: cosine similarity
    """
    norm_e1 = e1 / np.array([np.linalg.norm(e1)])
    norm_e2 = e2 / np.array([np.linalg.norm(e2)])

    matrix_sim = norm_e1.dot(norm_e2.T)
    return matrix_sim


def get_neighbourhoods_mean_sim(e1, e2, k=10):
    """
    Get the mean similarity of an embedding e1 (or list of embeddings) to its target neighbourhood in the space of e2, as well as the reverse
    :param e1: embeddings vector 1
    :param e2: embeddings vector 2
    :param k: size of neighbourhood, as defined in https://arxiv.org/pdf/1710.04087.pdf
    :return:
    """
    matrix_sim = get_matrix_cos_sim(e1, e2)
    return np.mean(np.sort(matrix_sim, axis=1)[:, -k:], axis=1), np.mean(np.sort(matrix_sim.T, axis=1)[:, -k:], axis=1)


def csls_bulk(e_1, e_space_2, r_e1, r_space_2):
    """
    The CSLS similarity function, applied in bulk for an embedding e_1 and another embedding space e_space_2
    :param e_1: embedding vector
    :param e_space_2: list of embedding vectors, denoting the second embedding space
    :param r_e1: mean similarity of embedding vector 2
    :param r_space_2: list of mean similarities of embeddings in vector space 2
    :return:
    """
    return (2 * get_matrix_cos_sim(e_1, e_space_2) - r_e1 - r_space_2)