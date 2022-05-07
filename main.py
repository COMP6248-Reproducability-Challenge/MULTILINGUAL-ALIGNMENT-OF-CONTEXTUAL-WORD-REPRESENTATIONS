# load data
import numpy as np
import tensorflow_datasets as tfds
from parse_europarl_data import create_parallel_sentences

# BERT wrappers
from BaseBertWrapper import BaseBertWrapper

# alignment procedure
from align_pretrained_embeddings import align_pretrained_embeddings

# evaluations
from evaluation import compute_word_retrieval_acc
from xnli_pipeline import xnli_pipeline

if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)
    wrapper_aligned = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)

    languages = ["bg"]
    tokens_files = ["data/data/europarl-v7.bg-en.token.clean.reverse"]
    alignment_files = ["data/data/europarl-v7.bg-en.intersect.reverse"]

    data = create_parallel_sentences(tokens_files, alignment_files, num_sentences=np.inf)
    num_test = 0
    num_dev = 100
    num_train = 200
    train = []
    dev = []
    test = []
    for d in data:
        test.append((d[0][:num_test], d[1][:num_test], d[2][:num_test]))
        dev.append((d[0][num_test:num_dev], d[1][num_test:num_dev], d[2][num_test:num_dev]))
        train.append((d[0][num_dev:num_train], d[1][num_dev:num_train], d[2][num_dev:num_train]))

    wrapper_aligned = align_pretrained_embeddings(wrapper_aligned, wrapper, train, languages, num_sent_train=100)

    for language, lang_data in zip(languages, dev):
        compute_word_retrieval_acc(wrapper_aligned, language, lang_data)

    xnli_data = tfds.load(name='xnli', split='test')
    mlni_data = tfds.load(name='multi_nli',split='train[:80000]')
    xnli_pipeline(wrapper, mlni_data, xnli_data)
    xnli_pipeline(wrapper_aligned, mlni_data, xnli_data)
