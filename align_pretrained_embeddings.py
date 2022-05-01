from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from parse_europarl_data import displace_alignments


def L(emb1, emb2):
    """
    Loss function for aligning pretrained embeddings, as defined in section 3.3
    :param emb1: f(i, s) -> in paper
    :param emb2: f(j, t) -> in paper
    :return:
    """
    return F.mse_loss(emb1, emb2)


def R(emb, emb_base):
    """
    Regularization term, as defined in section 3.3
    :param emb: f(i, t)
    :param emb_base: f0(i, t)
    :return:
    """
    return F.mse_loss(emb, emb_base)


def align_pretrained_embeddings(wrapper, base_wrapper, data, languages, lambda_reg=1, num_sent_train=10000):
    """
    Align pre-trained contextual embeddings
    """

    base_wrapper.bert.eval()
    num_epochs = 1
    batch_size = 2

    # Adam optimiser with values specified in Appendix A.1
    optimiser = torch.optim.Adam([param for param in wrapper.bert.parameters() if param.requires_grad], lr=0.00005,
                                 betas=(0.9, 0.98), eps=1e-9)

    max_num_sentences = max([len(d[0]) for d in data])
    if num_sent_train is None or num_sent_train > max_num_sentences:
        print("Number of train sentences was either unspecified or larger than the number of sentences available for at least a language, default to max number of sentences found")
        num_sent_train = max_num_sentences

    warmup_steps = int(num_sent_train * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=warmup_steps, num_training_steps=len(data) * num_epochs)

    # displace alignments beforehand
    # feature_indices = []
    # for idl, language in enumerate(languages):
    #     print(f"Recomputing aligning {language}")
    #     idx_features_1, idx_features_2 = displace_alignments(data[idl])
    #     feature_indices.append((idx_features_1, idx_features_2))

    for epoch in range(num_epochs):
        for b in tqdm(range(0, num_sent_train, batch_size)):
            loss_batch = None
            wrapper.train()

            # get from all languages
            for idl, language in enumerate(languages):
                sentences_1, sentences_2, alignments = data[idl]
                batch_s1, batch_s2, batch_alignments = sentences_1[b:b+batch_size], sentences_2[b:b+batch_size], alignments[b:b+batch_size]

                batch_idx1, batch_idx2 = displace_alignments((batch_s1, batch_s2, batch_alignments))

                feature_datal1 = wrapper.get_bert_data(batch_s1)
                feature_datal2 = wrapper.get_bert_data(batch_s2)
                feature_datal2_base = base_wrapper.get_bert_data(batch_s2)

                if loss_batch is None:
                    loss_batch = L(feature_datal1[batch_idx1], feature_datal2[batch_idx2]) + lambda_reg * R(feature_datal2, feature_datal2_base)
                else:
                    loss_batch += L(feature_datal1[batch_idx1], feature_datal2[batch_idx2]) + lambda_reg * R(feature_datal2, feature_datal2_base)

            loss_batch.backward()
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()

    return wrapper