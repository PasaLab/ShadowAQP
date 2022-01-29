import math
import os
import time
import numpy as np
import pandas as pd
import threading
import torch
from keras.utils.np_utils import to_categorical
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.utils.data import DataLoader
from torch import optim
from utils.model_utils import save_torch_model, load_torch_model
import logging
from utils.dataset_utils import *
from utils.pytorchtools import EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CVAE(nn.Module):
    def __init__(self, data_dim, label_dim, intermediate_dim, latent_dim, dataset):
        super(CVAE, self).__init__()
        self.device = dataset.device
        self.numeric_flag=len(dataset.numeric_columns)>0
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        # self.fc1 = nn.Linear(origin_dim, intermediate_dim)
        encoder_dim = [intermediate_dim]
        decoder_dim = [intermediate_dim]
        # encoder_dim = [intermediate_dim, intermediate_dim]
        # decoder_dim = [intermediate_dim, intermediate_dim]
        seq = []
        dim = data_dim + label_dim
        for item in encoder_dim:
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.encode_seq = Sequential(*seq)
        self.fc21 = nn.Linear(intermediate_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(intermediate_dim, latent_dim)  # var
        seq = []
        dim = latent_dim + label_dim
        for item in decoder_dim:
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.decode_seq = Sequential(*seq)
        # self.fc3 = nn.Linear(latent_dim, intermediate_dim)
        self.fc4 = nn.Linear(intermediate_dim, data_dim)
        self.output_info = dataset.encoded_output_info
        # self.output_layers = nn.ModuleList(
        #     [nn.Linear(intermediate_dim, digit) for digit, activ in dataset.encoded_output_info])
        # self.output_activ = [info[1] for info in dataset.encoded_output_info]

        if self.numeric_flag:
            self.sigma = nn.Parameter(torch.ones(data_dim + label_dim) * 0.1)
        else:
            self.sigma = []

    def encode(self, x, c):
        # h1 = F.relu(self.fc1(x))
        inputs = torch.cat([x, c], 1)
        h1 = self.encode_seq(inputs)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, c):
        # h3 = F.relu(self.fc3(z))
        inputs = torch.cat([z, c], 1)
        h3 = self.decode_seq(inputs)
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
        sigma=self.sigma
        if self.numeric_flag:
            sigma = self.sigma.clamp(-3, 3)  # p_params['logvar_x'] = self.logvar_x

        # sigma = Parameter(torch.ones(self.origin_dim) * 0.1)
        return output, sigma

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        return self.decode(z, c), mu, logvar  # 解码，同时输出均值方差

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
                loss_list.append(
                    F.cross_entropy(recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
            else:
                # loss_list.append(((x[:, st:ed] - recon_x[:, st:ed]) ** 2 / 2 / (sigma[st] ** 2)).sum())
                loss_list.append(((x[:, st:ed] - torch.tanh(recon_x[:, st:ed])) ** 2 / 2 / (sigma[st] ** 2)).sum())
                # loss_list.append(torch.log(sigma[st]) * x.size()[0])
            st = ed
        # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return sum(loss_list) * 2 / x.size()[0], kld / x.size()[0]


def torch_cvae_train(model, dataset, epochs, batch_size):
    start_time = time.perf_counter()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(),lr=1e-3)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    latent_param = {}
    for epoch in range(epochs):
        epoch_start_time = time.perf_counter()
        train_loss_vae, train_nll_vae, train_z_kld_vae = 3 * [0]
        for batch_idx, input_data in enumerate(loader):
            optimizer.zero_grad()
            x, c = input_data
            recon_x, mu, logvar = model(x, c)
            recon_loss, kld = model.loss_function(recon_x, x, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optimizer.step()
            # model.sigma.data.clamp_(0.01, 1.0)
            train_loss_vae += loss.item()
            train_nll_vae += recon_loss.item()
            train_z_kld_vae += kld.item()

        epoch_end_time = time.perf_counter()
        logger.info('----------------------------No.{} epoch----------------------------'.format(epoch + 1))
        logger.info('loss:{}, recon_loss:{}, kld_loss:{}, epoch_train_time:{}'.format(train_loss_vae,
                                                                                      train_nll_vae,
                                                                                      train_z_kld_vae,
                                                                                      (
                                                                                              epoch_end_time - epoch_start_time)))
        # early_stopping(loss, model)
        # # 若满足 early stopping 要求
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     # 结束模型训练
        #     break
        # if epoch + 1 == epochs:
        #     latent_param['mean'] = mu.detach()
        #     latent_param['std'] = logvar.mul(0.5).exp_().detach()

    end_time = time.perf_counter()
    logger.info('training time elapsed:{}'.format(end_time - start_time))
    return model


def generate_samples(model, dataset, query_config, train_config):
    sample_rate = train_config["sample_rate"]
    if train_config['sample_method'] == "senate":
        sample_allocation, sample_rates = senate_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "house":
        sample_allocation, sample_rates = house_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "advance sencate":
        sample_allocation, sample_rates = advance_senate_sampling(model, dataset, sample_rate)
    elif train_config['sample_method'] == "statistics":
        sample_allocation, sample_rates = statistics_sampling(model, dataset, sample_rate, query_config)
    else:
        sample_allocation, sample_rates = statistics_sampling_with_small_group(model, dataset, sample_rate,
                                                                               query_config)
    # print(sample_allocation)
    if 'condition' in query_config and len(query_config['condition']):
        logger.info("filtering with condition {}".format(query_config['condition']))
        condition=query_config['condition'][0]
        if '<=' in condition:
            bound_value=condition.split('<=')[-1]
            sample_allocation={k:v for k,v in sample_allocation.items() if k<=bound_value}
        elif '>=' in condition:
            bound_value = condition.split('>=')[-1]
            sample_allocation = {k: v for k, v in sample_allocation.items() if k >= bound_value}
        elif '=' in condition:
            bound_value = condition.split('=')[-1]
            sample_allocation = {k: v for k, v in sample_allocation.items() if k == bound_value}
        elif '<' in condition:
            bound_value = condition.split('<')[-1]
            sample_allocation = {k: v for k, v in sample_allocation.items() if k < bound_value}
        elif '>' in condition:
            bound_value = condition.split('>')[-1]
            sample_allocation = {k: v for k, v in sample_allocation.items() if k > bound_value}

    samples = generate_samples_with_allocation(dataset, model, sample_allocation, sample_rates, train_config)
    if 'outliers' in train_config and train_config['outliers'] == 'true':
        samples = pd.concat([samples, dataset.outliers])
    save_samples(samples, train_config)
    return samples


def load_torch_cvae(param):
    start_time = time.perf_counter()
    dataset = load_light_dataset(param)
    logger.info("feature info:{}".format(dataset.feature_info))
    latent_dim = param["latent_dim"]
    intermediate_dim = param["intermediate_dim"]
    model = CVAE(dataset.numeric_digits + dataset.categorical_digits, dataset.label_size, intermediate_dim, latent_dim,
                 dataset)
    model = load_torch_model(param,model)
    # model.to(dataset.device)
    if model is None:
        logger.error("model file not found")
        return
    end_time = time.perf_counter()
    logger.info("load model time elapsed:{}".format(end_time - start_time))
    return model, dataset


def train_torch_cvae(param):
    # hyper parameters
    start_time = time.perf_counter()
    lr = param["lr"]
    optimizer_type = param["optimizer_type"]
    batch_size = param["batch_size"]
    latent_dim = param["latent_dim"]
    intermediate_dim = param["intermediate_dim"]
    epochs = param["epochs"]
    logger.info("epoch:{}".format(epochs))
    logger.info("batch size:{}".format(batch_size))
    logger.info("latent dimension:{}".format(latent_dim))
    logger.info("intermediate dimension:{}".format(intermediate_dim))
    logger.info("gpu num:{}".format(param['gpu_num']))
    if exist_dataset(param):
        dataset = load_dataset(param)
    else:
        dataset = TabularDataset(param)
    logger.info("feature info:{}".format(dataset.feature_info))
    _, data_dim = dataset.data.shape
    model = CVAE(data_dim, dataset.label_size, intermediate_dim, latent_dim, dataset)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(dataset.device)
    model = torch_cvae_train(model, dataset, epochs=epochs, batch_size=batch_size)
    save_torch_model(model, param)
    save_dataset(dataset, param)
    end_time = time.perf_counter()
    logger.info("train model time elapsed:{}".format(end_time - start_time))
    return model, dataset


def generate_group_samples(sample_count, label, latent_dim, batch_size, model, z_decoded_list):
    start_time = time.perf_counter()
    while sample_count > 0:
        each_step_samples = sample_count if sample_count < batch_size else batch_size
        each_label = label[:each_step_samples, ]
        sample_count -= batch_size
        ### 0-1 normal
        mean = torch.zeros(each_step_samples, latent_dim).to(model.device)
        std = mean + 1

        ### param normal
        # mean = latent_param['mean']
        # std = latent_param['std']
        # mean = mean[:each_step_samples, ]
        # std = std[:each_step_samples, ]
        noise = torch.normal(mean=mean, std=std).to(model.device)
        fake, sigmas = model.decode(noise, each_label)
        column_list = []
        st = 0
        for digit, activ in model.output_info:
            ed = st + digit
            if activ == 'tanh':
                column_list.append(torch.tanh(fake[:, st:ed]))
            elif activ == 'softmax':
                column_list.append(torch.softmax(fake[:, st:ed], dim=1))
            elif activ == 'sigmoid':
                column_list.append(torch.sigmoid(fake[:, st:ed]))
            st = ed
        fake = torch.cat(column_list, dim=1)

        z_decoded = fake.detach().cpu().numpy()
        z_decoded_list.append(z_decoded)
    end_time = time.perf_counter()
    # logger.info('generate group samples time:{}'.format(end_time - start_time))


def generate_samples_with_allocation(dataset, model, sample_allocation, sample_rates,
                                     train_config):
    start_time = time.perf_counter()
    batch_size = train_config["batch_size"]
    latent_dim = train_config["latent_dim"]
    categorical_encoding = train_config["categorical_encoding"]
    z_decoded = []
    label_value_mapping = dataset.label_value_mapping
    label_size = len(label_value_mapping)

    for label_value_idx, label_value in label_value_mapping.items():
        sample_count = sample_allocation[label_value]
        if categorical_encoding == 'binary':
            mapping = dataset.label_mapping_out
            label = [mapping.loc[label_value_idx].values]
            label = torch.from_numpy(np.repeat(label, batch_size, axis=0)).to(model.device)
            # label = np.tile(label, (sample_count, 1))
        else:
            label = np.ones((batch_size,)) * label_value_idx
            label = torch.from_numpy(to_categorical(label, label_size)).to(model.device)

        # thread = threading.Thread(target=generate_group_samples,
        #                           args=(sample_count, label, latent_dim, batch_size, model, dataset, samples))
        # threads.append(thread)
        # thread.start()
        generate_group_samples(sample_count, label, latent_dim, batch_size, model, z_decoded)

    # for t in threads:
    #     t.join()
    z_decoded = np.concatenate(z_decoded, axis=0)
    samples_df = dataset.decode_samples(z_decoded)
    samples_df['{}_rate'.format(dataset.name)] = sample_rates
    # samples_df = pd.concat(samples)
    end_time = time.perf_counter()
    logger.info('sampling time:{}'.format(end_time - start_time))
    return samples_df


def house_sampling(model, dataset, sample_rate):
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    sample_rates = []
    sample_allocation = {}
    logger.info("house sampling rate:{}".format(sample_rate))
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = round(label_count * sample_rate)
        sample_allocation[label_value] = sample_count
        sample_rates += [sample_count / label_count] * sample_count
    return sample_allocation, sample_rates


def senate_sampling(model, dataset, sample_rate):
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate
    group_nums = len(label_group_counts)
    each_group_samples = int(total_samples / group_nums) + 1
    logger.info("senate sampling rate:{}".format(sample_rate))
    sample_allocation = {}
    sample_rates = []
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = each_group_samples if each_group_samples < label_count else label_count
        sample_allocation[label_value] = sample_count
        sample_rates += [sample_count / label_count] * sample_count
    return sample_allocation, sample_rates


# split the sample num into two part, one for senate, one for house
def advance_senate_sampling(model, dataset, sample_rate):
    model.eval()
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    total_rows = dataset.total_rows
    group_nums = len(label_group_counts)
    left_out_rate = 0.7
    total_samples = total_rows * sample_rate
    each_group_samples = math.ceil(total_samples * left_out_rate / group_nums)
    rest = math.ceil(total_samples * (1 - left_out_rate))
    logger.info("advance senate sampling rate:{}".format(sample_rate))
    label_sample_counts = {}
    small_group_total_rows = 0
    for (label_value, count) in label_group_counts.items():
        if count > each_group_samples:
            label_sample_counts[label_value] = each_group_samples
        else:
            label_sample_counts[label_value] = label_group_counts[label_value]
            rest += each_group_samples - label_sample_counts[label_value]
            small_group_total_rows += label_group_counts[label_value]
    big_group_total_rows = total_rows - small_group_total_rows
    for (label_value, count) in label_group_counts.items():
        if count > each_group_samples:
            label_sample_counts[label_value] += math.ceil(
                rest * (label_group_counts[label_value] / big_group_total_rows))

    sample_allocation = {}
    sample_rates = []
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        sample_count = label_sample_counts[label_value] if label_sample_counts[
                                                               label_value] < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates += [sample_count / group_count] * sample_count

    return sample_allocation, sample_rates


def statistics_sampling(model, dataset, sample_rate, query_config):
    model.eval()
    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate
    logger.info("statistics sampling rate:{}".format(sample_rate))
    sample_allocation = {}
    sample_rates = []
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        relative_variances = sum([label_group_relative_stds[col][label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(total_samples * (relative_variances / sum_relative_variance))
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates += [sample_count / group_count] * sample_count
    # logger.info("statistics sampling allocation:{}".format(sample_allocation))
    return sample_allocation, sample_rates


def statistics_sampling_with_small_group(model, dataset, sample_rate, query_config):
    model.eval()
    numeric_columns = list(set(query_config['sum_cols']) & set(query_config['avg_cols']) & set(dataset.numeric_columns))

    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_group_relative_stds = dataset.label_group_relative_stds
    label_group_relative_stds_sums = dataset.label_group_relative_stds_sums

    total_rows = dataset.total_rows
    total_samples = total_rows * sample_rate

    statistics_sampling_samples = total_samples * 0.5
    small_group_sampling_samples = total_samples - statistics_sampling_samples
    small_group_K = small_group_sampling_samples / len(label_group_counts)

    sample_allocation = {}
    sample_rates = []
    for label_value_idx, label_value in label_value_mapping.items():
        group_count = label_group_counts[label_value]
        relative_variances = sum([label_group_relative_stds[col][label_value] for col in numeric_columns])
        sum_relative_variance = sum([label_group_relative_stds_sums[col] for col in numeric_columns])
        group_sample = int(statistics_sampling_samples * (relative_variances / sum_relative_variance))
        group_sample += small_group_K
        sample_count = group_sample if group_sample < group_count else group_count
        sample_allocation[label_value] = sample_count
        sample_rates += [sample_count / group_count] * sample_count

    return sample_allocation, sample_rates


def save_samples(samples, param):
    samples_name = "{}_{}_{}_ld{}_id{}_bs{}_ep{}_rate{}_{}_{}.csv".format(param["model_type"], param["name"],
                                                                          '_'.join(param["label_columns"]),
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
