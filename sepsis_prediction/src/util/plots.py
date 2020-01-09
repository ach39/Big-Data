import matplotlib.pyplot as plt
from util.config import FIG_DIR
import fnmatch
import numpy as np
from other_models.ac_util import plot_confusion_matrix

import logging

logger = logging.getLogger(__name__)

def save_fig(fig, prefix, dir=FIG_DIR):
    pattern = '*' + prefix + '-*'
    count = 0
    for entry in dir.iterdir():
        logger.debug(entry)
        if fnmatch.fnmatch(entry, pattern):
            count += 1
            logger.debug(f"count = {count}")

    fig.savefig(FIG_DIR / f"{prefix}-{count}")


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    fig, ax = plt.subplots(2)
    ax[0].set_title('Loss Curve')
    ax[0].plot(np.arange(len(train_losses)), train_losses, label='Train')
    ax[0].plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('epoch')
    ax[0].legend(loc="best")
    ax[1].set_title('Accuracies')
    ax[1].plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
    ax[1].plot(np.arange(len(valid_losses)), valid_accuracies, label='Validation')
    ax[1].legend(loc="best")
    save_fig(fig, 'learning_curve')


def plot_confusion_matrix(results, class_names):
    '''
	Borrowed from
	https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
	:param results: A list of tuples (true, pred)
	:param class_names:  A list of all the class names
	:return:
	'''
    from sklearn.metrics import confusion_matrix

    res = list(zip(*results))
    y_true = [class_names[i] for i in res[0]]
    y_pred = [class_names[j] for j in res[1]]

    # print(results)
    # print(class_names)
    cm = confusion_matrix(y_true, y_pred, class_names)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_fig(fig, 'confusion')


if __name__ == "__main__":
    import numpy as np

    plot_learning_curves(np.random.rand(100), np.random.rand(100), np.random.rand(100), np.random.rand(100))
    plot_confusion_matrix([(0, 1), (2, 3), (1, 2), (0, 0), (3, 3)], [0, 1, 2, 3])
