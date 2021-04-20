from src.utils import *
from src.config import *
from src.data import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import copy
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from collections import Counter


def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight.data)


def get_criterion():
    return nn.CrossEntropyLoss()


def get_optimizer(model, lr, eps, weight_decay):
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=eps,
        weight_decay=weight_decay
    )


def train(model, device, dataloaders, criterion, optimizer, num_epochs):
    # set seed
    set_seed()

    # starting time
    since = time.time()

    # store model that yields the highest accuracy in the validation set
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    # keep track of the statistics
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    # iterate through each epoch
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()   # set model to evaluate mode

            # loss and number of correct predictions
            running_loss = 0.0
            running_corrects = 0
            running_data = 0

            # iterate over each batch
            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                for batch in tepoch:
                    # Unravel inputs
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # reset the parameter gradients
                    optimizer.zero_grad()
                    model.zero_grad()

                    # forward
                    # track history only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        outputs = outputs.to(device)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, axis=1)

                        # backward only if in training
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # add to statistics
                    running_loss += loss.item() * len(labels)
                    running_corrects += torch.sum(
                        preds == labels.data.flatten()
                    )
                    running_data += len(labels)

                    # update progress bar
                    tepoch.set_postfix(
                        loss=(running_loss / running_data),
                        accuracy=(running_corrects.item() / running_data)
                    )
                    time.sleep(0.1)

            # compute loss and accuracy at epoch level
            epoch_loss = running_loss / running_data
            epoch_acc = running_corrects.double() / running_data
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)

            print('{} Loss: {:.4f}; Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            # deep copy the model when epoch accuracy (on validation set) is the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                torch.save(model, 'model_checkpoint.pt')
                print('Best model so far! Saved checkpoint.')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60
    ))
    print('Best Val Acc: {:4f} at Epoch: {:d}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch, losses, accuracies


def predict_on_dev(model, device, dev_features, n, k):
    df = pd.DataFrame(columns=['wsi', 'label', 'vote_0',
                               'vote_1', 'vote_2', 'pred', 'correct'])
    i = 0
    for label, flabel in enumerate(['LUAD', 'LUSC', 'MESO']):
        for wsi, patches in dev_features[flabel].items():
            counts = predict(model, device, list(patches.values()), n, k)
            pred = counts.most_common()[0][0]
            df.loc[i] = [wsi, label,
                         counts[0], counts[1], counts[2],
                         pred, label == pred]
            i += 1
    return df


def predict(model, device, patches, n, k):
    model.eval()
    features = get_random_n_sets_of_k_patches(patches, n, k, False)
    padded_features = pad_features(features, k)
    inputs = torch.DoubleTensor(padded_features).to(device)
    outputs = model(inputs)
    preds = torch.argmax(outputs, axis=1)
    counts = Counter(preds.cpu().numpy())
    return counts
