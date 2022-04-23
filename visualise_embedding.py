import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from BaseBertWrapper import BaseBertWrapper


def get_bert_tokens(sentence, tokenizer):
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_word_representations(hidden_states):
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    token_vecs_sum = []
    for token in token_embeddings:
        sum_vec = torch.sum(token[-4:], dim=0).numpy()
        token_vecs_sum.append(sum_vec)

    return token_vecs_sum


def get_word_vectors(sentence, wrapper):
    tokenized_text, tokens_tensor, segments_tensors = get_bert_tokens(sentence, wrapper.tokenizer)
    outputs = wrapper.bert(tokens_tensor, segments_tensors)
    hidden_states = outputs['hidden_states']
    word_vecs = get_word_representations(hidden_states)

    # remove the BERT tokens
    return tokenized_text[1:-1], word_vecs[1:-1]


def visualise_words(tsne_data, datasets):
    plt.figure(figsize=(8, 8))
    colors = cm.rainbow(np.linspace(0, 1, len(datasets)))
    markers = ["o", "s", "D"]

    tsne_model = TSNE(n_components=2, init='pca', n_iter=2500, random_state=23)
    transformed_values = tsne_model.fit_transform(tsne_data)

    for ds_index, (tokens, vectors) in enumerate(datasets):

        # TODO: some sort of function that finds the transformed values from the tsne output
        new_values = []

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        for i in range(len(x)):
            plt.scatter(x[i], y[i], color=colors[ds_index], marker=markers[ds_index])
            plt.annotate(tokens[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    plt.show()


if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)
    wrapper.bert.eval()

    texts = ["bank",
             "The river bank was flooded.",
             "The bank vault was robust.",
             "He had to bank on her for support.",
             "The bank was out of money.",
             "The bank teller was a man."]

    datasets = []

    with torch.no_grad():
        labels, vectors = [], []

        for idx, text in enumerate(texts):
            tokens, word_vecs = get_word_vectors("[CLS] " + text + " [SEP]", wrapper)

            labels.append(f"bank-{idx}")
            vectors.append(word_vecs[tokens.index('bank')])

        datasets.append((labels, vectors))

    visualise_words(
        datasets
    )