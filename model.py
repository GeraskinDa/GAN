import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from time import time
from torch.optim.lr_scheduler import StepLR
from IPython.display import clear_output
from utils import draw_history, draw_images
from torchvision import transforms
from PIL import Image


def conv_block(in_ch, out_ch, activation='relu', *args, **kwargs):

    activations = nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['lrelu', nn.LeakyReLU(0.1)]
    ])

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, *args, **kwargs),
        nn.InstanceNorm2d(out_ch),
        activations[activation],
        nn.Conv2d(out_ch, out_ch, *args, **kwargs),
        nn.InstanceNorm2d(out_ch),
        activations[activation]
    )


class Generator(nn.Module):

    def __init__(self, sizes, *args, **kwargs):
        super().__init__()
        self.sizes = sizes

        self.down_convs = nn.ModuleList([conv_block(in_ch, out_ch, kernel_size=3, padding=1, *args, **kwargs)
                                         for in_ch, out_ch in zip(sizes, sizes[1:])])

        self.pools = nn.ModuleList([nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2)
                                    for in_ch in sizes[1:]])

        self.bottleneck = conv_block(sizes[-1], 2 * sizes[-1], kernel_size=3, padding=1)

        self.upsamples = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                                        for in_ch, out_ch in zip(2 * np.array(sizes[::-1]),
                                                                 sizes[:0:-1])])

        self.up_convs = nn.ModuleList([conv_block(in_ch, out_ch, kernel_size=3, padding=1, *args, **kwargs)
                                       for in_ch, out_ch in zip(2 * np.array(sizes[::-1]), sizes[:0:-1])])

        self.last = nn.Conv2d(sizes[1], sizes[0], kernel_size=3, padding=1)

    def forward(self, x):

        self.skips = []
        for conv, pool in zip(self.down_convs, self.pools):
            x = conv(x)
            self.skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for i, (conv, ups) in enumerate(zip(self.up_convs, self.upsamples)):
            x = ups(x)
            x = conv(torch.cat([self.skips[-(i + 1)], x], dim=1))

        x = torch.tanh(self.last(x))

        return x


class Discriminator(nn.Module):

    def __init__(self, sizes, *args, **kwargs):
        super().__init__()

        self.sizes = sizes
        self.down_convs = nn.ModuleList(
            [conv_block(in_ch, out_ch, activation='lrelu', kernel_size=3, padding=1, *args, **kwargs)
             for in_ch, out_ch in zip(sizes, sizes[1:])])

        self.pools = nn.ModuleList([nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2)
                                    for in_ch in sizes[1:]])

        self.last = nn.Conv2d(sizes[-1], 1, kernel_size=3, padding=1)

    def forward(self, x):
        for conv, pool in zip(self.down_convs, self.pools):
            x = pool(conv(x))

        x = self.last(x)
        x = torch.sigmoid(x)

        return x


def update_discriminator(disc, optimizer, loss_fn, data_batch_reals, data_batch_fakes, device):
    optimizer.zero_grad()

    preds_reals = disc(data_batch_reals)
    reals_labels = torch.full(preds_reals.size(), fill_value=1.).to(device)
    loss_on_reals = loss_fn(preds_reals, reals_labels)

    preds_fakes = disc(data_batch_fakes)
    fakes_labels = torch.full(preds_fakes.size(), fill_value=0.).to(device)
    loss_on_fakes = loss_fn(preds_fakes, fakes_labels)

    loss = (loss_on_reals + loss_on_fakes) / 2.

    loss.backward()
    optimizer.step()
    accuracy_on_reals = np.mean((preds_reals.cpu().detach().numpy() > 0.5) == reals_labels.cpu().numpy())
    accuracy_on_fakes = np.mean((preds_fakes.cpu().detach().numpy() > 0.5) == fakes_labels.cpu().numpy())

    return loss_on_reals.cpu().item(), accuracy_on_reals, loss_on_fakes.cpu().item(), accuracy_on_fakes


def set_parameters_req_grad(model, req_grad=True):
    for params in model.parameters():
        params.required_grad = req_grad


def update_pool(pool, images, max_size=50):
    buffer = []

    for img in images:
        img = img.cpu().detach().numpy()
        if len(pool) < max_size:
            pool.append(img)
            buffer.append(img)
        elif random.random() > 0.5:
            ix = random.randint(0, len(pool) - 1)
            out_img = pool[ix]
            pool[ix] = img
            buffer.append(out_img)
        else:
            buffer.append(img)

    return torch.tensor(buffer)


def train_cycle_model(gen_class, dis_class, train_dataloader, weights, epoch_num, path_to_save, device):
    gens_loss_history = []
    dis_x_loss_history = []
    dis_y_loss_history = []
    dis_x_reals_acc_history = []
    dis_x_fakes_acc_history = []
    dis_y_reals_acc_history = []
    dis_y_fakes_acc_history = []

    pool_gen_x2y = []
    pool_gen_y2x = []

    batch_num = len(train_dataloader)

    x2y_params = [3, 32, 64, 128, 256]
    y2x_params = [3, 32, 64, 128, 256]

    gen_x2y = gen_class(x2y_params, padding_mode='reflect').to(device)
    gen_y2x = gen_class(y2x_params, padding_mode='reflect').to(device)

    dis_x_params = [3, 8, 16, 32]
    dis_y_params = [3, 8, 16, 32]

    dis_x = dis_class(dis_x_params).to(device)
    dis_y = dis_class(dis_y_params).to(device)

    l1loss = torch.nn.L1Loss()
    mseloss = torch.nn.MSELoss()

    gens_optimizer = torch.optim.Adam(list(gen_x2y.parameters()) + list(gen_y2x.parameters()),
                                      amsgrad=True)

    scheduler = StepLR(gens_optimizer, step_size=15, gamma=0.5)

    dis_x_optimizer = torch.optim.Adam(dis_x.parameters(), amsgrad=True)
    dis_y_optimizer = torch.optim.Adam(dis_y.parameters(), amsgrad=True)

    random.seed(0)
    samples_to_draw = next(iter(train_dataloader))

    for epoch in range(epoch_num):
        start = time()

        gen_x2y.train()
        gen_y2x.train()
        dis_x.train()
        dis_y.train()

        running_loss_gens = 0
        running_loss_dis_x = 0
        running_loss_dis_y = 0
        running_reals_acc_dis_x = 0
        running_fakes_acc_dis_x = 0
        running_reals_acc_dis_y = 0
        running_fakes_acc_dis_y = 0

        for batch_data in train_dataloader:
            data_X = batch_data['X'].to(device)
            data_Y = batch_data['Y'].to(device)

            # ---------------------#
            # Updating generators #
            # ---------------------#
            gens_optimizer.zero_grad()

            # To save memory
            set_parameters_req_grad(dis_x, req_grad=False)
            set_parameters_req_grad(dis_y, req_grad=False)

            gen_x2y_out = gen_x2y(data_X)
            cycle_data_X = gen_y2x(gen_x2y_out)

            gen_y2x_out = gen_y2x(data_Y)
            cycle_data_Y = gen_x2y(gen_y2x_out)

            # Cycle consistency loss
            cycle_loss = l1loss(data_X, cycle_data_X) + l1loss(data_Y, cycle_data_Y)

            # Identity loss
            identity_loss = l1loss(data_Y, gen_x2y(data_Y)) + l1loss(data_X, gen_y2x(data_X))

            # Adversarial loss
            pred_dis_x = dis_x(gen_y2x_out)
            pred_dis_y = dis_y(gen_x2y_out)

            X_labels = torch.full(size=pred_dis_x.size(), fill_value=1.).to(device)
            Y_labels = torch.full(size=pred_dis_y.size(), fill_value=1.).to(device)

            adv_loss = mseloss(pred_dis_x, X_labels) + mseloss(pred_dis_y, Y_labels)

            # Final loss for generators
            loss = weights[0] * adv_loss + weights[1] * identity_loss + weights[2] * cycle_loss
            loss.backward()

            running_loss_gens += loss.cpu().item()

            gens_optimizer.step()

            # -------------------------#
            # Updating discriminators #
            # -------------------------#

            # Optimize discriminator X

            set_parameters_req_grad(dis_x)
            set_parameters_req_grad(dis_y)

            gen_x2y_out = update_pool(pool_gen_x2y, gen_x2y_out).to(device)
            gen_y2x_out = update_pool(pool_gen_y2x, gen_y2x_out).to(device)

            # gen_x2y_out.detach_()
            # gen_y2x_out.detach_()

            loss_reals, acc_reals, loss_fakes, acc_fakes = update_discriminator(dis_x, dis_x_optimizer,
                                                                                mseloss, data_X, gen_y2x_out, device)

            running_loss_dis_x += (loss_reals + loss_fakes) / 2.
            running_reals_acc_dis_x += acc_reals
            running_fakes_acc_dis_x += acc_fakes

            # Optimize disciminator Y

            loss_reals, acc_reals, loss_fakes, acc_fakes = update_discriminator(dis_y, dis_y_optimizer,
                                                                                mseloss, data_Y, gen_x2y_out, device)

            running_loss_dis_y += (loss_reals + loss_fakes) / 2.
            running_reals_acc_dis_y += acc_reals
            running_fakes_acc_dis_y += acc_fakes

        gens_loss_history.append(running_loss_gens / batch_num)
        dis_x_loss_history.append(running_loss_dis_x / batch_num)
        dis_y_loss_history.append(running_loss_dis_y / batch_num)
        dis_x_reals_acc_history.append(running_reals_acc_dis_x / batch_num)
        dis_x_fakes_acc_history.append(running_fakes_acc_dis_x / batch_num)
        dis_y_reals_acc_history.append(running_reals_acc_dis_y / batch_num)
        dis_y_fakes_acc_history.append(running_fakes_acc_dis_y / batch_num)

        scheduler.step()

        end = time()

        clear_output(wait=True)
        draw_images(gen_x2y, gen_y2x, samples_to_draw, device, epoch, end - start)

        draw_history([gens_loss_history, dis_x_loss_history, dis_y_loss_history,
                      dis_x_reals_acc_history, dis_x_fakes_acc_history,
                      dis_y_reals_acc_history, dis_y_fakes_acc_history])
        plt.show()

        torch.save({'gen_x2y': {'state_dict': gen_x2y.state_dict(),
                                'params': x2y_params},
                    'gen_y2x': {'state_dict': gen_y2x.state_dict(),
                                'params': y2x_params},
                    'dis_x': {'state_dict': dis_x.state_dict(),
                              'params': dis_x_params},
                    'dis_y': {'state_dict': dis_y.state_dict(),
                              'params': dis_y_params},
                    'gens_optimizer': gens_optimizer.state_dict(),
                    'dis_x_optimizer': dis_x_optimizer.state_dict(),
                    'dis_y_optimizer': dis_y_optimizer.state_dict(),
                    'epoch_num': epoch,
                    'history': {'gens_loss': gens_loss_history,
                                'dis_x_loss': dis_x_loss_history,
                                'dis_y_loss': dis_y_loss_history,
                                'dis_x_reals_acc': dis_x_reals_acc_history,
                                'dis_x_fakes_acc': dis_x_fakes_acc_history,
                                'dis_y_reals_acc': dis_y_reals_acc_history,
                                'dis_y_fakes_acc': dis_y_fakes_acc_history}}, path_to_save)


def test_generators(gen_class, path, dataloader, device):
    checkpoint = torch.load(path)

    gen_x2y = gen_class(checkpoint['gen_x2y']['params']).to(device)
    gen_y2x = gen_class(checkpoint['gen_y2x']['params']).to(device)

    gen_x2y.load_state_dict(checkpoint['gen_x2y']['state_dict'])
    gen_y2x.load_state_dict(checkpoint['gen_y2x']['state_dict'])

    samples_to_draw = next(iter(dataloader))

    draw_images(gen_x2y, gen_y2x, samples_to_draw, device)


def test_one_sample(gen_class, generator_type, path, sample_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    checkpoint = torch.load(path)

    resize = transforms.Resize((256, 256))

    sample = Image.open(sample_path).convert('RGB')
    sample_to_show = resize(sample)
    sample_in = torch.unsqueeze(transform(sample), 0).to(device)

    if generator_type == 'x2y':
        generator = gen_class(checkpoint['gen_x2y']['params']).to(device)
        generator.load_state_dict(checkpoint['gen_x2y']['state_dict'])
    elif generator_type == 'y2x':
        generator = gen_class(checkpoint['gen_y2x']['params']).to(device)
        generator.load_state_dict(checkpoint['gen_y2x']['state_dict'])

    generator.eval()

    sample_out = (torch.squeeze(generator(sample_in).cpu().detach(), dim=0).numpy() + 1.) / 2.

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(np.array(sample_to_show))
    axes[0].set_title('original')

    axes[1].imshow(np.transpose(sample_out, axes=(1, 2, 0)))
    axes[1].set_title('generated')


def show_history(path, device):

    checkpoint = torch.load(path, map_location=device)
    history = []
    for item in checkpoint['history'].items():
        history.append(item[1])

    draw_history(history)