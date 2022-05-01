# load data
from parse_europarl_data import create_parallel_sentences

from BaseBertWrapper import BaseBertWrapper
from MultiLingualAlignedBert import MultiLingualAlignedBertWrapper

from align_pretrained_embeddings import align_pretrained_embeddings

from evaluation import compute_word_retrieval_acc

if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False)
    wrapper_aligned = MultiLingualAlignedBertWrapper('bert-base-multilingual-cased', do_lower_case=False)

    languages = ["bg"]
    tokens_files = ["data/data/europarl-v7.bg-en.token.clean.reverse"]
    alignment_files = ["data/data/europarl-v7.bg-en.intersect.reverse"]

    data = create_parallel_sentences(tokens_files, alignment_files, num_sentences=2048)

    # for language, lang_data in zip(languages, data):
    #     compute_word_retrieval_acc(wrapper, language, lang_data)

    wrapper_aligned = align_pretrained_embeddings(wrapper_aligned, wrapper, data, languages)

    print()


