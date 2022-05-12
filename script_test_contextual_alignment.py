# load data
import json
import torch

from parse_europarl_data import create_parallel_sentences

# BERT wrappers
from BaseBertWrapper import BaseBertWrapper

# evaluations
from evaluation import compute_word_retrieval_acc


if __name__ == "__main__":
    wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False, cuda=True)
    wrapper.bert.load_state_dict(torch.load("checkpoint_bert10000.pth"))

    accuracies = []
    languages = ["bg", "de", "el", "es", "fr"]
    # take one language at a time for memory reasons
    for lan in languages:
        tokens_files = [f"data/data/europarl-v7.{lan}-en.token.clean.reverse"]
        alignment_files = [f"data/data/europarl-v7.{lan}-en.intersect.reverse"]

        data = create_parallel_sentences(tokens_files, alignment_files, num_sentences=5000)
        num_test = 1024
        test = []
        for d in data:
            test.append((d[0][:num_test], d[1][:num_test], d[2][:num_test]))

        acc_source_target, acc_target_source = compute_word_retrieval_acc(wrapper, lan, test[0])
        accuracies.append(
            {
                "language": lan,
                "contextual": True,
                "num_test": num_test,
                "acc1": acc_source_target,
                "acc2": acc_target_source
            }
        )
        del data, test

    with open('results/random_runs/data_10k.json', 'w') as f:
        json.dump(accuracies, f)
