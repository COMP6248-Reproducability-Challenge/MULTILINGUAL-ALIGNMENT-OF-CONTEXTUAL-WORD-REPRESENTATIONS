import torch
import numpy as np
from tqdm import tqdm

from align_pretrained_embeddings import L
from loss_functions import get_neighbourhoods_mean_sim, csls_bulk
from parse_europarl_data import create_parallel_sentences, displace_alignments

# BERT wrappers
from BaseBertWrapper import BaseBertWrapper

if __name__ == "__main__":
    wrapper_types = [False, True]
    languages = ["bg", "de", "el", "es", "fr"]
    # take one language at a time for memory reasons
    for lan in languages:
        print(f"BEGINNING LANGUAGE {lan}")
        tokens_files = [f"data/data/europarl-v7.{lan}-en.token.clean.reverse"]
        alignment_files = [f"data/data/europarl-v7.{lan}-en.intersect.reverse"]

        data = create_parallel_sentences(tokens_files, alignment_files, num_sentences=2000)
        num_test = 1024
        test = []
        for d in data:
            test.append((d[0][:num_test], d[1][:num_test], d[2][:num_test]))
        idx_features_1, idx_features_2 = displace_alignments(test[0])

        embeddings = []
        lg_types = [lan, "en"]
        i = 0
        for wrapper_type in wrapper_types:
            wrapper = BaseBertWrapper('bert-base-multilingual-cased', do_lower_case=False, cuda=True, pytorch_pretrained=wrapper_type)

            wrapper.eval()
            with torch.no_grad():
                feature_datal1 = wrapper(test[0][0]).detach().cpu().numpy()[idx_features_1]
                feature_datal2 = wrapper(test[0][1]).detach().cpu().numpy()[idx_features_2]
                embeddings.append([feature_datal1, feature_datal2])


        # now we have embeddings in the spaces of the both pretrained models
        for e_l1_m1, e_l1_m2 in zip(embeddings[0], embeddings[1]):
            neigh_e1, neigh_e2 = get_neighbourhoods_mean_sim(e_l1_m1, e_l1_m2)

            matches_e1 = np.array([])
            matches_e2 = np.array([])
            distances_csls_1 = np.array([])
            distances_csls_2 = np.array([])
            mses = np.array([])

            # for each word
            for idw in tqdm(range(len(e_l1_m1))):
                w1 = e_l1_m1[idw]
                w2 = e_l1_m2[idw]

                # get distancces from both languages to eachother
                distances_l1_to_l2 = csls_bulk(w1, e_l1_m2, neigh_e1[idw], neigh_e2)
                distances_l2_to_l1 = csls_bulk(w2, e_l1_m1, neigh_e2[idw], neigh_e1)
                distances_csls_1 = np.append(distances_csls_1, distances_l1_to_l2[idw])
                distances_csls_2 = np.append(distances_csls_2, distances_l2_to_l1[idw])

                # get mses
                mses = np.append(mses, L(torch.Tensor(w1), torch.Tensor(w2)).detach().cpu().numpy())

                matches_e1 = np.append(matches_e1, int(np.argsort(-distances_l1_to_l2)[0] == idw))
                matches_e2 = np.append(matches_e2, int(np.argsort(-distances_l2_to_l1)[0] == idw))

            print(f"Accuracy from Transformers to Pytorch on {lg_types[i]}: {(1 / len(matches_e1)) * np.sum(matches_e1)}")
            print(f"Accuracy from Pytorch to Transformers on {lg_types[i]}: {(1 / len(matches_e2)) * np.sum(matches_e2)}")
            print(f"Average distance Transformers to Pytorch on {lg_types[i]}: {np.mean(distances_csls_1)}")
            print(f"Average distance Pytorch to Transformers on {lg_types[i]}: {np.mean(distances_csls_2)}")
            print(f"Average MSE distance Transformers, Pytorch on {lg_types[i]} : {np.mean(mses)}")
            i += 1


