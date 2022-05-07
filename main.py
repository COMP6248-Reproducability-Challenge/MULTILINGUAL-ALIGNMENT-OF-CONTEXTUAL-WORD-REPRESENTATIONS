# load data
from parse_europarl_data import create_parallel_sentences

# BERT wrappers
from BaseBertWrapper import BaseBertWrapper

# alignment procedure
from align_pretrained_embeddings import align_pretrained_embeddings

# evaluations
from evaluation import compute_word_retrieval_acc

if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)
    wrapper_aligned = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)

    languages = ["bg"]
    tokens_files = ["data/data/europarl-v7.bg-en.token.clean.reverse"]
    alignment_files = ["data/data/europarl-v7.bg-en.intersect.reverse"]

    data = create_parallel_sentences(tokens_files, alignment_files, num_sentences=200)

    num_dev = 100
    train = []
    dev = []
    for d in data:
        dev.append((d[0][:num_dev], d[1][:num_dev], d[2][:num_dev]))
        train.append((d[0][num_dev:], d[1][num_dev:], d[2][num_dev:]))

    wrapper_aligned = align_pretrained_embeddings(wrapper_aligned, wrapper, train, languages)

    for language, lang_data in zip(languages, dev):
        compute_word_retrieval_acc(wrapper_aligned, language, lang_data)


