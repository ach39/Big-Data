# import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util.config import *
from rnn.mydatasets import VisitSequenceWithLabelDataset, visit_collate_fn
from rnn.lstm import MyLSTM
from util.plots import plot_learning_curves, save_fig
from util.utils import train, evaluate
from other_models.ac_util import plot_confusion_matrix, calc_scores

logger = logging.getLogger(__name__)

NUM_EPOCHS = 100
BATCH_SIZE = 8
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 1
criterion = nn.CrossEntropyLoss()

train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

def test_dataset():
    return VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)

def test_loader():
    return DataLoader(dataset=test_dataset(), batch_size=1, shuffle=False, collate_fn=visit_collate_fn,
                      num_workers=NUM_WORKERS)


def predict_mortality(model, device, data_loader):
    model.eval()
    probas = []
    # reference: https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            out = model.forward(inputs)
            proba = torch.max(out).item()
            proba = min(proba, 1)
            proba = max(proba, 0)
            probas.append(proba)

        print(probas)
        return probas

def num_features():
    return len(train_seqs[0][0])


def train_variable_rnn():
    logger.info("Epochs: {}, batch size: {}, Cuda: {}, workers: {}".format(NUM_EPOCHS, BATCH_SIZE,
                                                                           USE_CUDA, NUM_WORKERS))

    torch.manual_seed(1)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loading
    print('===> Loading entire datasets')

    assert (len(train_seqs) == len(train_labels))
    assert (len(valid_seqs) == len(valid_labels))
    assert (len(test_seqs) == len(test_labels))

    logger.info("num_features: {}".format(num_features()))
    logger.info("train patients: {}".format(len(train_seqs)))
    logger.info("val patients: {}".format(len(valid_seqs)))

    train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features())
    valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features())


    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn,
                              num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn,
                              num_workers=NUM_WORKERS)
    # batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs

    model = MyLSTM(batch_size=BATCH_SIZE, input_size=num_features())

    optimizer = optim.Adam(model.parameters())

    model.to(device)
    criterion.to(device)

    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    for epoch in range(NUM_EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(model, MODEL_PATH)

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)


def score_model(path=MODEL_PATH):
    best_model = torch.load(path)
    test_loss, test_accuracy, test_results = evaluate(best_model, device, test_loader(), criterion)

    class_names = ['No Sepsis', 'Sepsis']
    # plot_confusion_matrix(test_results, class_names)

    Y = []
    y_pred = []

    for i,j in test_results:
        Y.append(i)
        y_pred.append(j)

    calc_scores(Y, y_pred, clf_name="LSTM")
    fig = plot_confusion_matrix(Y, y_pred, clf_name="LSTM")
    save_fig(fig, "confusion")


if __name__ == "__main__":
    train_variable_rnn()
    score_model()
