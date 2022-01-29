import math
import time
import logging
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import losses
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils.model_utils import save_keras_model, load_keras_model
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import concatenate as concat
from keras.callbacks import EarlyStopping
import pandas as pd
from utils.dataset_utils import TabularDataset
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon


def train_cvae(param):
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
    # split train and test
    dataset = TabularDataset(param)
    data = dataset.data
    feature_info = dataset.feature_info
    n_row, n_col = data.shape
    origin_dim = n_col
    logger.info("origin dimension{}".format(origin_dim))
    ### one column label
    # label_column_name = param["label_columns"]
    # label_group_counts = dataset.label_group_counts

    ### multiple column label
    # label_column_name = "-".join(param['label_columns'])
    # data[label_column_name] = data[param['label_columns']].agg('-'.join, axis=1)
    # dataset.origin_df[label_column_name] = data[param['label_columns']].astype(str).sum(axis=1)
    # label_group_counts = dataset.label_group_counts

    label_column_name = dataset.label_column_name
    label_group_counts = dataset.label_group_counts

    ## split the data into training data and test data
    # split_index = int(n_row * 0.7)
    # x_train = data.iloc[0:split_index, :].values
    # y_train = data.iloc[0:split_index, data.columns.str.contains(label_column)].values
    # x_test = data.iloc[split_index:n_row, :].values
    # y_test = data.iloc[split_index:n_row, data.columns.str.contains(label_column)].values

    ### use all the data as training data and 30% samples of the data as test data
    x_train = data.values
    y_train = data.iloc[:, data.columns.str.contains(label_column_name)].values
    test_data = data  # .sample(frac=0.3)
    x_test = test_data.values
    y_test = test_data.iloc[:, data.columns.str.contains(label_column_name)].values
    # y_train = to_categorical(data.iloc[0:split_index, label_column].values)
    # y_test = to_categorical(data.iloc[split_index:n_row, label_column].values)
    label_size = y_train.shape[1]

    # build encoder layer
    x = Input(shape=(origin_dim,))
    label = Input(shape=(label_size,))
    inputs = concat([x, label])
    encoder_h1 = Dense(intermediate_dim, activation='relu')(inputs)
    # encoder_h2 = Dense(intermediate_dim, activation='relu')(encoder_h1)
    z_mean = Dense(latent_dim, activation='linear')(encoder_h1)
    z_log_var = Dense(latent_dim, activation='linear')(encoder_h1)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    zc = concat([z, label])

    # build decoder layer
    decoder_h = Dense(intermediate_dim, activation='relu')
    # decoder_h2 = Dense(intermediate_dim, activation='relu')
    h_decoded1 = decoder_h(zc)
    # h_decoded = decoder_h2(h_decoded1)
    output_info = dataset.encoded_output_info

    # if activate_type == "softmax":
    #     # use sigmoid function for all columns
    #     decoder_mean = Dense(origin_dim, activation='sigmoid')
    #     x_decoded_mean = decoder_mean(h_decoded1)
    # else:
    # use different activ function for different columns
    out_layer_list = [Dense(digit, activation=activ) for digit, activ in output_info]
    # out_layer_list = [Dense(col_domain_size, activation='softmax') if col_type == "categorical"
    #                   else Dense(col_domain_size)
    #                   for col_name, col_type, col_domain_size in feature_info]
    out_list = []
    for out_layer in out_layer_list:
        out_list.append(out_layer(h_decoded1))
    x_decoded_mean = concat(out_list)

    vae = Model([x, label], x_decoded_mean)

    # define loss
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    if loss_agg_type == "sum":
        loss_agg_fun = K.sum
    else:
        loss_agg_fun = K.mean

    # if loss_type == "bce":
    #     reconstruct_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
    #     vae_loss = loss_agg_fun(reconstruct_loss + kl_loss)
    # else:
    numeric_column_len = len(dataset.numeric_columns)
    reconstruct_loss_num = losses.mean_squared_error(x[:, :numeric_column_len],
                                                     x_decoded_mean[:, :numeric_column_len])
    # reconstruct_loss_cat = K.categorical_crossentropy(x[:, numeric_column_len:], x_decoded_mean[:, numeric_column_len:])
    reconstruct_loss_cat = tf.zeros_like(reconstruct_loss_num)

    st = 0
    for digit, activ in output_info:
        ed = st + digit
        if activ == 'softmax':
            reconstruct_loss_cat += K.categorical_crossentropy(x[:, st:ed], x_decoded_mean[:, st:ed])
        elif activ == 'sigmoid':
            reconstruct_loss_cat += K.binary_crossentropy(x[:, st:ed], x_decoded_mean[:, st:ed])
        elif activ == 'tanh':
            reconstruct_loss_cat += losses.mean_squared_error(x[:, st:ed], x_decoded_mean[:, st:ed])
        st = ed
    # categorical_index = numeric_column_len
    # for (col_name, col_type, col_domain_size) in feature_info:
    #     reconstruct_loss_cat += K.categorical_crossentropy(
    #         x[:, categorical_index:categorical_index + col_domain_size],
    #         x_decoded_mean[:, categorical_index:categorical_index + col_domain_size])
    #     categorical_index += col_domain_size
    vae_loss = loss_agg_fun(reconstruct_loss_num + reconstruct_loss_cat + kl_loss)

    # build vae models
    vae.add_loss(vae_loss)
    optimizer = Adam(lr=lr)
    if optimizer_type == "rmsp":
        optimizer = RMSprop(lr=lr)
    logger.info("optimizer:{}".format(optimizer_type))
    vae.compile(optimizer=optimizer)
    vae.summary()

    # train models
    if train_flag == "train":
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
        vae.fit([x_train, y_train],
                shuffle=False,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([x_test, y_test], None),
                callbacks=[early_stopping])
        # vae.fit([x_train, y_train],
        #         shuffle=True,
        #         epochs=epochs,
        #         batch_size=batch_size)
        save_keras_model(vae, param)
    else:
        load_keras_model(vae, param)

    model_weight = vae.get_weights()
    # encoder = Model([x,label], z_mean)
    decoder_input = Input(shape=(latent_dim + label_size,))
    _h_decoded1 = decoder_h(decoder_input)
    # _h_decoded = decoder_h2(_h_decoded1)
    # if activate_type == "softmax":
    #     _x_decoded_mean = decoder_mean(_h_decoded1)
    # else:
    _out_list = []
    for out_layer in out_layer_list:
        _out_list.append(out_layer(_h_decoded1))
    _x_decoded_mean = concat(_out_list)
    generator = Model(decoder_input, _x_decoded_mean)

    # generate samples
    z_decoded = generate_samples(generator, sample_rate, model_weight, latent_dim, batch_size, dataset,
                                 param["categorical_encoding"])  # , binary_encoder=dataset.bce)
    samples = dataset.decode_samples(z_decoded)
    save_samples(samples, param)
    return samples


# def generate_sample(generator, sample_rate, latent_dim, label_column_name, label_group_counts,
#                     batch_size, binary_encoder=None):
#     start_time = time.perf_counter()
#     # label_count = [int(total_samples / label_range) for label_num in range(label_range)]
#     for each in label_group_counts:
#         label_group_counts[each] = round(label_group_counts[each] * sample_rate)
#     mapping = binary_encoder.mapping
#     for label_num in range(len(label_group_counts)):
#         # label = to_categorical(label_num, label_range).reshape(1, -1)
#         label = mapping[label_column_name].loc[label_num].values
#         # label=label.repeat(label_group_counts[label_num],axis=1)
#         label = np.tile(label, (label_group_counts[label_num], 1))
#         # label = to_categorical(np.full((label_group_counts[label_num], 1), label_num), label_range)
#         label_samples = []
#         for i in range(latent_dim):
#             z_sample = np.random.normal(z_mean_w[i], z_std_w[i], label_group_counts[label_num])
#             label_samples.append(z_sample)
#         label_samples = np.array(label_samples).T
#         label_samples = np.column_stack([label_samples, label])
#         z_samples.append(label_samples)
#     z_samples = np.concatenate(z_samples, axis=0)
#     z_decoded = generator.predict(z_samples, batch_size=batch_size)
#     end_time = time.perf_counter()
#     logger.info('[INFO]decode(decoder) samples time:{}'.format(end_time - start_time))
#     return z_decoded


def generate_samples(generator, sample_rate, model_weights, latent_dim, batch_size, dataset, categorical_encoding):
    start_time = time.perf_counter()
    z_mean_w = model_weights[3]
    z_std_w = [math.exp(logvar * 0.5) for logvar in model_weights[5]]
    z_samples = []
    label_group_counts = dataset.label_group_counts
    label_value_mapping = dataset.label_value_mapping
    label_size = len(label_group_counts)
    # label_count = [int(total_samples / label_range) for label_num in range(label_range)]
    for each in label_group_counts:
        label_group_counts[each] = round(label_group_counts[each] * sample_rate)
    # mapping = binary_encoder.mapping
    for label_value_idx, label_value in label_value_mapping.items():
        label_count = label_group_counts[label_value]
        sample_count = round(label_count * sample_rate)
        if categorical_encoding == 'binary':
            mapping = dataset.label_mapping_out
            label = mapping.loc[label_value_idx].values
            label = np.repeat(label, batch_size, axis=0)
        else:
            label = np.ones((label_count,)) * label_value_idx
            label = to_categorical(label, label_size)
        label_samples = []
        while sample_count > 0:
            each_step_samples = sample_count if sample_count < batch_size else batch_size
            each_label = label[:each_step_samples, ]
            sample_count -= batch_size
            noise = np.random.normal(loc=0, scale=1, size=(each_step_samples, latent_dim))
            z_decoded = generator.predict(noise, batch_size=batch_size)
            z_samples.append(z_decoded)
        # for i in range(latent_dim):
        #     z_sample = np.random.normal(z_mean_w[i], z_std_w[i], label_count)
        #     label_samples.append(z_sample)
        # label_samples = np.array(label_samples).T
        # label_samples = np.column_stack([label_samples, label])
        # z_samples.append(label_samples)
    z_samples = np.concatenate(z_samples, axis=0)
    # z_decoded = generator.predict(z_samples, batch_size=batch_size)
    end_time = time.perf_counter()
    logger.info('decode(decoder) samples time:{}'.format(end_time - start_time))
    return z_samples


def save_samples(samples, param):
    samples_name = "samples.csv"
    if param["model_type"] == "keras_vae":
        samples_name = "{}_{}_ld{}_id{}_bs{}_ep{}.csv".format(param["model_type"], param["name"], param["latent_dim"],
                                                              param["intermediate_dim"], param["batch_size"],
                                                              param["epochs"])
    elif param["model_type"] == "keras_cvae":
        samples_name = "{}_{}_{}_ld{}_id{}_bs{}_ep{}.csv".format(param["model_type"], param["name"],
                                                                 '_'.join(param["label_columns"]),
                                                                 param["latent_dim"],
                                                                 param["intermediate_dim"], param["batch_size"],
                                                                 param["epochs"])
    samples.to_csv("./output/{}".format(samples_name), index=False)
