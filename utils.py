import numpy as np
import torch
import matplotlib.pyplot as plt
import random


def back_normalize(images):
    """Makes each image have values ranging from 0 to 1

    Parameters
    ----------
    images: list
        List of batches of torch.tensors with values in the range
        of -1 to -1

    Returns
    -------
    images: list
        List of numpy arrays
    """

    for i, img in enumerate(images):
        images[i] = np.transpose((img.cpu().numpy() + 1.) / 2., axes=(0, 2, 3, 1))

    return images


def draw_images(gen_x2y, gen_y2x, samples_to_draw, device, epoch=None, time=None):
    """Draws 2 images per domain, the outputs of generators
    and recovered data each epoch

    Parameters
    ----------

    gen_x2y: Generator class object
        Generator from domain X to Y
    gen_y2x: Generator class object
        Generator from domain Y to X
    samples_to_draw: dict
        Samples (from train_dataloader) to draw
    device: device
    epoch: int
        Current epoch number
    time: int
        Time for one epoch
    """
    samples_X = samples_to_draw['X']
    samples_Y = samples_to_draw['Y']

    samples_X = samples_X[:2].to(device)
    samples_Y = samples_Y[:2].to(device)

    gen_x2y.eval()
    gen_y2x.eval()

    with torch.no_grad():
        pred_y = gen_x2y(samples_X)
        pred_x = gen_y2x(samples_Y)

        cycle_x = gen_y2x(pred_y)
        cycle_y = gen_x2y(pred_x)

    images = back_normalize([samples_X, samples_Y, pred_y, pred_x, cycle_x, cycle_y])

    num = len(samples_X)
    titles = ['Input', 'Output', 'Restored']

    if time is not None:
        time = int(time)
        minutes, seconds = time // 60, time % 60
    else:
        minutes, seconds = None, None

    fig, axes = plt.subplots(3, 2 * num, figsize=(20, 15))
    fig.suptitle(f"Epoch is {epoch}\nTime to train last epoch {minutes}.{seconds}m")

    for i in range(3):
        for j in range(num):
            axes[i][j].imshow(images[2 * i][j])
            axes[i][j].set_title(titles[i])

    for i in range(3):
        for j in range(num):
            axes[i][j + num].imshow(images[2 * i + 1][j])
            axes[i][j + num].set_title(titles[i])


def make_ax(ax, history, label, xlabel, ylabel, title):
    """Configures axis parameters

    Parameters
    ----------
    ax: matplotlib.axes.Axes class object
    history: list
        History of training to draw
    label: str
    xlabel: str
        Title for x axis
    ylabel: str
        Title for y axis
    title: str

    """
    ax.plot(history, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def draw_history(history):
    """Draws history of training

    Parameters
    ----------
    history: list
        List containing loss and metric histories for each models

    """
    [gens_loss_history, dis_x_loss_history, dis_y_loss_history,
     dis_x_reals_acc_history, dis_x_fakes_acc_history,
     dis_y_reals_acc_history, dis_y_fakes_acc_history] = history

    fig, axes = plt.subplots(4, figsize=(13, 40))

    make_ax(axes[0], gens_loss_history, 'Generators loss', 'epoch number', 'loss', 'Loss of models: training')
    make_ax(axes[1], dis_x_loss_history, 'Discriminator X loss', 'epoch number', 'loss', 'Loss of models: training')
    make_ax(axes[1], dis_y_loss_history, 'Discriminator Y loss', 'epoch number', 'loss', 'Loss models: training')

    make_ax(axes[2], dis_x_reals_acc_history, 'accuracy on reals', 'epoch number',
            'mean accuracy', 'Accuracy of discriminator X: training')
    make_ax(axes[2], dis_x_fakes_acc_history, 'accuracy on fakes', 'epoch number',
            'mean accuracy', 'Accuracy of discriminator X: training')
    make_ax(axes[3], dis_y_reals_acc_history, 'accuracy on reals', 'epoch number',
            'mean accuracy', 'Accuracy of discriminator Y: training')
    make_ax(axes[3], dis_y_fakes_acc_history, 'accuracy on fakes', 'epoch number',
            'mean accuracy', 'Accuracy of discriminator Y: training')


def set_seed(seed=0):
    """Sets seed of each random_fn to be equal to the seed

    Parameters
    ----------
    seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


