import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from BaseBertWrapper import BaseBertWrapper
from parse_europarl_data import create_parallel_sentences, displace_alignments

fig = 0

def tsne_data(data):
    tsne_model = TSNE(n_components=2, init='pca', n_iter=2500, random_state=42)
    return tsne_model.fit_transform(data)


def plot_data(datasets):
    global fig
    plt.figure(figsize=(8, 8))
    colors = ["r", "b"]
    markers = ["o", "v", "X", "s", "^"]

    for ds_index, dataset in enumerate(datasets):

        # for each embedding of a word
        for idw, (values, word) in enumerate(dataset):
            x = []
            y = []
            for coords in values:
                x.append(coords[0])
                y.append(coords[1])

            plt.scatter(x, y, color=colors[idw], marker=markers[ds_index], label=word)

    plt.legend()
    plt.savefig(f"word_embeddings_{fig}.png")
    fig += 1
    plt.show()


if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)
    wrapper_aligned = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)

    wrapper.eval()
    wrapper_aligned.eval()

    words = [("Jahr", "year"), ("wollte", "wanted"), ("Frage", "question"), ("Ich", "I"), ("Gelegenheit", "opportunity")]

    languages = ["de"]
    for lan in languages:
        tokens_files = [f"data/data/europarl-v7.{lan}-en.token.clean.reverse"]
        alignment_files = [f"data/data/europarl-v7.{lan}-en.intersect.reverse"]

        data = create_parallel_sentences(tokens_files, alignment_files, num_sentences=5000)[0]
        num_test = 1024
        data = (data[0][:num_test], data[1][:num_test], data[2][:num_test])

        embeddings_l1 = []
        embeddings_l2 = []

        idx_features_1, idx_features_2, w1, w2 = displace_alignments(data, return_words=True)

        with torch.no_grad():
            e1 = wrapper(data[0]).detach().cpu().numpy()[idx_features_1]
            e2 = wrapper(data[1]).detach().cpu().numpy()[idx_features_2]

            e1_aligned = wrapper_aligned(data[0]).detach().cpu().numpy()[idx_features_1]
            e2_aligned = wrapper_aligned(data[1]).detach().cpu().numpy()[idx_features_2]

            words_1 = np.array(w1)[idx_features_1]
            words_2 = np.array(w2)[idx_features_2]

        l1_tsne = tsne_data(np.concatenate(e1, e1_aligned))
        l2_tsne = tsne_data(np.concatenate(e2, e2_aligned))

        e1_tsne = l1_tsne[:len(e1)]
        e1_tsne_aligned = l1_tsne[len(e1):]

        e2_tsne = l2_tsne[:len(e2)]
        e2_tsne_aligned = l2_tsne[len(e2):]

        plotted_data = []
        plotted_data_algined = []

        for (word_1, word_2) in words:
            emb_w1 = e1_tsne[np.where(words_1 == word_1)[0]]
            emb_w2 = e2_tsne[np.where(words_2 == word_2)[0]]

            emb_w1_aligned = e1_tsne_aligned[np.where(words_1 == word_1)[0]]
            emb_w2_aligned = e2_tsne_aligned[np.where(words_2 == word_2)[0]]

            plotted_data.append( [
                    (emb_w1, word_1),
                    (emb_w2, word_2)
                ]
            )

            plotted_data_algined.append([
                (emb_w1_aligned, word_1),
                (emb_w2_aligned, word_2)
            ]
            )

        plot_data(plotted_data)
        plot_data(plotted_data_algined)
        print()