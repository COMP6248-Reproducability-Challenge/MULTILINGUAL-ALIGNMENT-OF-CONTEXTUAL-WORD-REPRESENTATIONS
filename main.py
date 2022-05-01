import torch

from BaseBertWrapper import BaseBertWrapper
from parse_europarl_data import create_parallel_sentences
from evaluation import compute_word_retrieval_acc

if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)

    tokens_files = ["data/data/europarl-v7.bg-en.token.clean.reverse"]
    alignment_files = ["data/data/europarl-v7.bg-en.intersect.reverse"]

    data = create_parallel_sentences(tokens_files, alignment_files)
    compute_word_retrieval_acc(wrapper, "bg", data)



    print()


