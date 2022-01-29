import os
import time
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.utils.data import DataLoader
from torch import optim
from utils.model_utils import save_torch_model, load_torch_model
import logging
from utils.dataset_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VAE(nn.Module):
    def __init__(self, data_dim, intermediate_dim, latent_dim, dataset):
        super(VAE, self).__init__()
        self.origin_dim = data_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        # self.fc1 = nn.Linear(origin_dim, intermediate_dim)
        encoder_dim = (intermediate_dim, intermediate_dim)
        decoder_dim = (intermediate_dim, intermediate_dim)
        seq = []
        dim = data_dim
        for item in list(encoder_dim):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.encode_seq = Sequential(*seq)
        self.fc21 = nn.Linear(intermediate_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(intermediate_dim, latent_dim)  # var
        seq = []
        dim = latent_dim
        for item in list(decoder_dim):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.decode_seq = Sequential(*seq)
        # self.fc3 = nn.Linear(latent_dim, intermediate_dim)
        self.fc4 = nn.Linear(intermediate_dim, data_dim)

        self.output_info = dataset.encoded_output_info
        self.output_layers = nn.ModuleList(
            [nn.Linear(intermediate_dim, digit) for digit, activ in dataset.encoded_output_info])
        self.output_activ = [info[1] for info in dataset.encoded_output_info]

        self.sigma = nn.Parameter(torch.ones(data_dim) * 0.1)

        # self.numeric_columns = dataset.numeric_columns
        # if len(self.numeric_columns) > 0:
        #     self.logvar_x = nn.Parameter(torch.ones(origin_dim).float() * 0.1)
        # else:
        #     self.logvar_x = []

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = self.encode_seq(x)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        h3 = self.decode_seq(z)
        output = self.fc4(h3)
        # output_list = []
        # for idx, layer in enumerate(self.output_layers):
        #     if self.output_activ[idx] == 'sigmoid':
        #         output_list.append(F.sigmoid(layer(h3)))
        #     elif self.output_activ[idx] == 'softmax':
        #         output_list.append(F.softmax(layer(h3)))
        #     else:
        #         output_list.append(layer(h3))
        #         # output_list.append(F.tanh(layer(h3)))
        # output = torch.cat(output_list, 1)
        # output = F.tanh(self.fc4(h3))
        # if self.numeric_columns:
        #     sigma = self.logvar_x.clamp(-3, 3)  # p_params['logvar_x'] = self.logvar_x

        # sigma = Parameter(torch.ones(self.origin_dim) * 0.1)
        return output, self.sigma

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        return self.decode(z), mu, logvar  # 解码，同时输出均值方差

    def loss_function(self, recon_x, x, mu, logvar):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        """

        sigma = recon_x[1]
        recon_x = recon_x[0]
        loss_list = []
        st = 0
        for digit, activ in self.output_info:
            ed = st + digit
            if activ == 'sigmoid':
                loss_list.append(F.binary_cross_entropy(torch.sigmoid(recon_x[:, st:ed]), x[:, st:ed], reduction='sum'))
            elif activ == 'softmax':
                loss_list.append(F.cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            else:
                # loss_list.append(((x[:, st:ed] - recon_x[:, st:ed]) ** 2 / 2 / (sigma[st] ** 2)).sum())
                loss_list.append(((x[:, st:ed] - torch.tanh(recon_x[:, st:ed])) ** 2 / 2 / (sigma[st] ** 2)).sum())
                loss_list.append(torch.log(sigma[st]) * x.size()[0])

            st = ed
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return sum(loss_list) * 2 / x.size()[0], kld / x.size()[0]


def train_torch_vae(param):
    # hyper parameters
    lr = param["lr"]
    optimizer_type = param["optimizer_type"]
    epochs = param["epochs"]
    batch_size = param["batch_size"]
    intermediate_dim = param["intermediate_dim"]
    latent_dim = param["latent_dim"]
    train_flag = param["train_flag"]
    loss_agg_type = param["loss_agg_type"]
    sample_rate = param["sample_rate"]

    if train_flag == "train":
        # split train and test
        start_time = time.perf_counter()
        if exist_dataset(param):
            dataset = load_dataset(param)
        else:
            dataset = TabularDataset(param)

        data = dataset.data
        n_row, n_col = data.shape
        origin_dim = n_col

        ## use all the data as training data and 30% samples of the data as test data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        model = VAE(origin_dim, intermediate_dim, latent_dim, dataset)
        optimizer = optim.Adam(model.parameters())
        # optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)

        for epoch in range(epochs):
            train_loss_vae, train_nll_vae, train_z_kld_vae = 3 * [0]
            for batch_idx, input_data in enumerate(loader):
                optimizer.zero_grad()
                # real = data[0].to(self._device)
                x, _ = input_data
                recon_x, mu, logvar = model(x)
                recon_loss, kld = model.loss_function(recon_x, x, mu, logvar)
                loss = recon_loss + kld
                loss.backward()
                optimizer.step()
                model.sigma.data.clamp_(0.01, 1.0)
                train_loss_vae += loss.item()
                train_nll_vae += recon_loss.item()
                train_z_kld_vae += kld.item()
            if epoch % 1 == 0:
                print('----------------------------No.{} epoch----------------------------'.format(epoch + 1))
                print('loss:{}, recon_loss:{}, kld_loss:{}'.format(train_loss_vae,
                                                                   train_nll_vae,
                                                                   train_z_kld_vae))
        save_torch_model(model, param)
        save_dataset(dataset, param)
        end_time = time.perf_counter()
        logger.info('training time elapsed:{}'.format(end_time - start_time))
    else:
        dataset = load_dataset(param)
        n_row = dataset.total_rows
        model = load_torch_model(param)
        if model is None:
            logger.error("model file not found")
            return

    samples = generate_samples(model, n_row, sample_rate, latent_dim, batch_size, dataset)
    # samples = dataset.decode_samples(z_decoded)
    save_samples(samples, param)
    return samples


def generate_samples(model, total_num, sample_rate, latent_dim, batch_size, dataset):
    start_time = time.perf_counter()
    model.eval()

    sample_count = round(total_num * sample_rate)
    steps = sample_count // batch_size + 1

    samples = []
    z_decoded_list = []
    while sample_count > 0:
        each_step_samples = sample_count if sample_count < batch_size else batch_size
        sample_count -= each_step_samples
        mean = torch.zeros(each_step_samples, latent_dim)
        std = mean + 1
        noise = torch.normal(mean=mean, std=std)  # .to(self._device)
        fake, sigmas = model.decode(noise)
        column_list = []
        st = 0
        for digit, activ in model.output_info:
            ed = st + digit
            if activ == 'tanh':
                column_list.append(torch.tanh(fake[:, st:ed]))
                # column_list.append(fake[:, st:ed])
            elif activ == 'softmax':
                column_list.append(torch.softmax(fake[:, st:ed], dim=1))
            elif activ == 'sigmoid':
                column_list.append(torch.sigmoid(fake[:, st:ed]))
            st = ed
        fake = torch.cat(column_list, dim=1)
        # fake = torch.tanh(fake)
        z_decoded = fake.detach().cpu().numpy()
        z_decoded_list.append(z_decoded)
        # z_samples = dataset.decode_samples(z_decoded)
        # samples.append(z_samples)
    z_decoded_list = np.concatenate(z_decoded_list, axis=0)
    samples_df = dataset.decode_samples(z_decoded_list)
    # samples_df = pd.concat(samples)
    samples_df['rate'] = sample_rate
    end_time = time.perf_counter()
    logger.info('generate sample time:{}'.format(end_time - start_time))
    return samples_df


def save_samples(samples, param):
    samples_name = "{}_{}_ld{}_id{}_bs{}_ep{}_rate{}_{}_{}.csv".format(param["model_type"], param["name"],
                                                                       param["latent_dim"],
                                                                       param["intermediate_dim"],
                                                                       param["batch_size"],
                                                                       param["epochs"], param["sample_rate"],
                                                                       param["categorical_encoding"],
                                                                       (param["numeric_encoding"] + str(
                                                                           param["max_clusters"])) if param[
                                                                                                          "numeric_encoding"] == 'gaussian' else
                                                                       param["numeric_encoding"])
    samples.to_csv("./output/{}".format(samples_name), index=False)
