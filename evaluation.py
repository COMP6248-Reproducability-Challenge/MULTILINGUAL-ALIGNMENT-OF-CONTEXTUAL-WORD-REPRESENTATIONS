import numpy as np

from parse_europarl_data import displace_alignments
from loss_functions import get_neighbourhoods_mean_sim, csls_bulk


def get_accuracy(source_emb, target_emb):
    if source_emb.shape != target_emb.shape:
        raise Exception("Different number of source and target embeddings")

    # compute the mean similarity of all embeddings in the source and target
    neigh_e1, neigh_e2 = get_neighbourhoods_mean_sim(source_emb, target_emb)

    matches_e1 = np.array([])
    matches_e2 = np.array([])
    # for each word
    for idw, (w1, w2) in enumerate(np.array(list(zip(source_emb, target_emb)))):
        matches_e1 = np.append(matches_e1, int( np.argmax(csls_bulk(w1, target_emb, neigh_e1[idw], neigh_e2)) == idw ))
        matches_e2 = np.append(matches_e2, int( np.argmax(csls_bulk(w2, source_emb, neigh_e2[idw], neigh_e1)) == idw ))

    return (1 / source_emb.shape[0]) * np.sum(matches_e1), (1 / source_emb.shape[0]) * np.sum(matches_e2)


def compute_word_retrieval_acc(wrapper, language_source, data, language_target="en"):
    print(f"Computing word retrieval accuracy for language {language_source}")
    wrapper.bert.eval()

    idx_features_1, idx_features_2 = displace_alignments(data)

    feature_datal1 = wrapper.get_bert_data(data[0]).detach().numpy()[idx_features_1]
    feature_datal2 = wrapper.get_bert_data(data[1]).detach().numpy()[idx_features_2]

    acc_source_target, acc_target_source = get_accuracy(feature_datal1, feature_datal2)
    print(f"Accuracy from {language_source} to {language_target}: {acc_source_target}")
    print(f"Accuracy from {language_target} to {language_source}: {acc_target_source}")


def compute_xnli_cross_lingual_transfer(dataset):
    return 0