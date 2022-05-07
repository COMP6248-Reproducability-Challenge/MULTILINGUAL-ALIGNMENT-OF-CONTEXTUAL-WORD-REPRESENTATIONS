import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from LinearClassifier import LinearClassifier
from load_xnli_data import load_prepare_nli


def train_classifier(clf, train_dataloader, use_cuda=False):
    preds = []
    true_labels = []
    loss_list = []
    num_epochs = 3

    criterion = nn.CrossEntropyLoss().to(torch.device('cuda') if use_cuda else torch.device('cpu'))
    optimiser = torch.optim.Adam([param for param in clf.parameters() if param.requires_grad], lr=0.00005,
                                 betas=(0.9, 0.98), eps=1e-9)

    warmup_steps = int(len(train_dataloader) * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps = warmup_steps, num_training_steps = len(train_dataloader)*num_epochs)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_dataloader):
            batch = tuple(b.to(torch.device('cuda') if use_cuda else torch.device('cpu')) for b in batch)

            b_input_ids, b_input_mask, b_segment, b_labels = batch

            optimiser.zero_grad()
            out = clf(b_input_ids)
            labels = b_labels
            loss = criterion(out, labels)

            loss.backward()
            scheduler.step()
            optimiser.step()

            preds.append(torch.argmax(out, dim=1))
            epoch_loss += loss.item()
            true_labels.append(labels)
            loss_list.append(loss.detach().cpu().numpy())
    return clf


def predict_classifier(clf, test_dataloader, use_cuda=False):
    preds = []
    true_labels = []

    clf.eval()

    for batch in tqdm(test_dataloader):
        batch = tuple(b.to(torch.device('cuda') if use_cuda else torch.device('cpu')) for b in batch)

        b_input_ids, b_input_mask, b_segment, b_labels = batch

        with torch.no_grad():
            # Logit Predictions made
            outputs = clf(b_input_ids)

        logits = torch.argmax(outputs, dim=1).to('cpu').numpy()
        labels = b_labels.to('cpu').numpy()

        preds.append(logits)
        true_labels.append(labels)

    flat_predictions = np.concatenate(preds, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)

    acc = (flat_predictions == flat_true_labels).mean()
    return acc


def xnli_pipeline(wrapper, data_train, data_test, use_cuda=False):

    # get data
    input_ids_ar, attn_masks_ar, segment_ids_ar, labels_ar = load_prepare_nli(wrapper, data_train)
    train_dataset = TensorDataset(input_ids_ar, attn_masks_ar, segment_ids_ar, labels_ar)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)

    # define linear classifier
    lc = LinearClassifier(wrapper.bert)
    lc.to(torch.device('cuda') if use_cuda else torch.device('cpu'))
    lc.train()

    # train the linear classifier
    lc = train_classifier(wrapper, lc, train_dataloader, use_cuda=False)

    input_ids_val, attn_masks_val, segment_ids_val, labels_val = load_prepare_nli(wrapper, data_test, language_code='en', language_index=4)
    test_dataset = TensorDataset(input_ids_val, attn_masks_val, segment_ids_val, labels_val)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    print(predict_classifier(lc, test_dataloader))







