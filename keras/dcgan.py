import os
import numpy as np
# For reproducibility 
np.random.seed(9999)
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, concatenate, Input
from keras.layers.core import Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D, AveragePooling1D
from keras.optimizers import SGD, adam
import argparse
import math

from sklearn import preprocessing
from sklearn.externals import joblib

from scipy.io import netcdf
#from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def generator_model(noise_dim=100, aux_dim=47, model_name="generator"):
    # Merge noise and auxilary inputs
    gen_input = Input(shape=(noise_dim,), name="noise_input")
    aux_input = Input(shape=(aux_dim,), name="auxilary_input")
    x = concatenate([gen_input, aux_input], axis=-1)

    # Dense Layer 1
    x = Dense(10 * 100)(x) 
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) # output shape is 10*100

    # Reshape the tensors to support CNNs
    x = Reshape((100, 10))(x) # shape is 100 x 10

    # Conv Layer 1
    x = Conv1D(filters=250, kernel_size=13, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) # output shape is 100 x 250
    x = UpSampling1D(size=2)(x) # output shape is 200 x 250

    # Conv Layer 2
    x = Conv1D(filters=100, kernel_size=13, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x) # output shape is 200 x 100
    x = UpSampling1D(size=2)(x) # output shape is 400 x 100
    
    # Conv Layer 3
    x = Conv1D(filters=1, kernel_size=13, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x) # final output shape is 400 x 1

    generator_model = Model(
        outputs=[x], inputs=[gen_input, aux_input], name=model_name)

    return generator_model


def discriminator_model(model_name="discriminator"):
    disc_input = Input(shape=(400, 1), name="discriminator_input")
    aux_input = Input(shape=(47,), name="auxilary_input")

    # Conv Layer 1
    x = Conv1D(filters=100, kernel_size=13, padding='same')(disc_input)
    x = LeakyReLU(0.2)(x) # output shape is 100 x 400
    x = AveragePooling1D(pool_size=20)(x) # ouput shape is 100 x 20

    # Conv Layer 2
    x = Conv1D(filters=250, kernel_size=13, padding='same')(x)
    x = LeakyReLU(0.2)(x) # output shape is 250 x 20
    x = AveragePooling1D(pool_size=5)(x) # output shape is 250 x 4

    # Conv Layer 3
    x = Conv1D(filters=300, kernel_size=13, padding='same')(x)
    x = LeakyReLU(0.2)(x) # output shape is 300 x 4
    x = Flatten()(x) # output shape is 1200

    x = concatenate([x, aux_input], axis=-1) # shape is 1247
    
    # Dense Layer 1
    x = Dense(200)(x)
    x = LeakyReLU(0.2)(x) # output shape is 200
    
    # Dense Layer 2
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    discriminator_model = Model(
        outputs=[x], inputs=[disc_input, aux_input], name=model_name)

    return discriminator_model


def generator_containing_discriminator(generator, discriminator):
    gen_input = Input(shape=(100,), name="noise_input")
    aux_input = Input(shape=(47,), name="auxilary_input") 
    x = generator([gen_input, aux_input]) # output shape is 400 x 1
    x = discriminator([x, aux_input]) # output shape is 1
    CGAN_model = Model(outputs=[x], inputs=[gen_input, aux_input], name="CGAN")
    return CGAN_model


def plot_feats(generated_feats, epoch, index):
    mean_feats = generated_feats.mean(axis=0)
    plt.figure()
    plt.plot(mean_feats)
    plt.savefig('figures/mean_pulse_epoch{}_index{}.png'.format(epoch, index))
    plt.close()


def train(BATCH_SIZE, X_train, Y_train):
    print X_train.shape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    print X_train.shape

    generator = generator_model()
    discriminator = discriminator_model()
    discriminator_on_generator = generator_containing_discriminator(
        generator, discriminator)

    d_optim = SGD(lr=0.0001, momentum=0.9, nesterov=True)
    g_optim = adam(lr=0.0001)

    generator.compile(loss='binary_crossentropy', optimizer="adam")

    discriminator.trainable = False
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    no_epochs = 17
    no_batches = int(X_train.shape[0] / BATCH_SIZE)
    noise_scale = 0.5
    noise = np.zeros((BATCH_SIZE, 100))

    for epoch in range(no_epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        ind = np.random.permutation(X_train.shape[0])
        X_train = X_train[ind]
        Y_train = Y_train[ind]
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            x_feats_batch = X_train[
                index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            y_feats_batch = Y_train[
                index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

            # generating noise for generator network
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.normal(
                    loc=0, scale=noise_scale, size=100)

            # embed the y feats on the noise
            generated_feats = generator.predict(
                [noise, y_feats_batch], verbose=0)

            if index % 200 == 0:
                feats = generated_feats.reshape(BATCH_SIZE, 400)
                plot_feats(feats, epoch, index)

            # embed the y feats on the real pulses
            d_loss_real = discriminator.train_on_batch(
                [x_feats_batch, y_feats_batch], [1] * BATCH_SIZE)
            # embed the y feats on the fake pulses
            d_loss_fake = discriminator.train_on_batch(
                [generated_feats, y_feats_batch], [0] * BATCH_SIZE)

            print("batch %d d_loss_real : %f, d_loss_fake : %f " %
                  (index, d_loss_real, d_loss_fake))

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.normal(
                    loc=0, scale=noise_scale, size=100)

            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                [noise, y_feats_batch], [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, Y_test,  nice=False, noise_scale=0.5):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE * 20, 100))
        for i in range(BATCH_SIZE * 20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_feats = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_feats, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_featss = np.zeros((BATCH_SIZE, 1) +
                               (generated_feats.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_featss[i, 0, :, :] = generated_feats[idx, 0, :, :]
        #feats = combine_feats(nice_featss)
    else:
        feats = np.zeros((Y_test.shape[0], 400))
        num_repeat = 1
        Y_test = np.repeat(Y_test, num_repeat, axis=0)
        noise = np.zeros((Y_test.shape[0], 100))
        for i in range(Y_test.shape[0]):
            noise[i, :] = np.random.normal(loc=0, scale=noise_scale, size=100)
        generated_feats = generator.predict([noise, Y_test], verbose=0)
	generated_feats = generated_feats.reshape(Y_test.shape[0], 400)
        j = 0
        for i in range(0, Y_test.shape[0], num_repeat):
            feats[j, :] = np.mean(generated_feats[i:i + num_repeat], axis=0)
            j = j + 1
    return feats


def load_data(data_dir, num_files=30):
    files_list = os.listdir(data_dir)
    data = None
    ac_data = None
    for fname in files_list[:num_files]:
        print fname
        f = os.path.join(data_dir, fname)
        with netcdf.netcdf_file(f, 'r') as fid:
            m = fid.variables['outputMeans'][:].copy()
            s = fid.variables['outputStdevs'][:].copy()
            feats = fid.variables['targetPatterns'][:].copy()
            ac_feats = fid.variables['inputs'][:].copy()
            scaler = preprocessing.StandardScaler()
            scaler.mean_ = m
            scaler.scale_ = s
            feats = scaler.inverse_transform(feats)
            assert feats.shape[0] == ac_feats.shape[0]
            # feats = np.concatenate((feats,ac_feats),axis=1)
        if data == None and ac_data == None:
            data = feats
            ac_data = ac_feats
        else:
            data = np.vstack((data, feats))
            ac_data = np.vstack((ac_data, ac_feats))
    return data, ac_data


def read_binary_file(file, dim=1):
    f = open(file, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    assert data.shape[0] % dim == 0.
    data = data.reshape(-1, dim)
    return data


def load_test_data(data_dir, mean_vec, std_vec):
    files_list = os.listdir(data_dir)
    ac_data = None
    for f in files_list:
        fname, ext = os.path.splitext(f)
        if ext == '.lf0':
            print fname
            lf0_file = os.path.join(data_dir, f)
            gain_file = os.path.join(data_dir, fname + '.gain')
            lsf_file = os.path.join(data_dir, fname + '.lsf')
            slsf_file = os.path.join(data_dir, fname + '.slsf')
            hnr_file = os.path.join(data_dir, fname + '.hnr')
            lf0_data = read_binary_file(lf0_file, dim=1)
            lsf_data = read_binary_file(lsf_file, dim=30)
            slsf_data = read_binary_file(slsf_file, dim=10)
            hnr_data = read_binary_file(hnr_file, dim=5)
            gain_data = read_binary_file(gain_file, dim=1)
            print lsf_data.shape, gain_data.shape, lf0_data.shape, hnr_data.shape, slsf_data.shape
            # [lsf gain lf0 hnr slsf]
            data = np.concatenate(
                (lsf_data, gain_data, lf0_data, hnr_data, slsf_data), axis=1)
            print data.shape
            scaler = preprocessing.StandardScaler()
            scaler.mean_ = mean_vec
            scaler.scale_ = std_vec
            data = scaler.transform(data)
            out_file = os.path.join(data_dir, fname + '.cmp')
            with open(out_file, 'w') as fid:
                data.tofile(fid)
    return ac_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--data_dir", type=str,
                        default="data/jenny")
    parser.add_argument("--testdata_dir", type=str,
                        default="tts_test_data")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        data, ac_data = load_data(args.data_dir, num_files=10)
	print np.max(data), np.min(data)
        train(BATCH_SIZE=args.batch_size, X_train=data,
              Y_train=ac_data)
    elif args.mode == "generate":
        
        test_dir = args.testdata_dir

        for f in os.listdir(test_dir):
            fname, ext = os.path.splitext(f)
            if ext == '.cmp':
                print fname
                cmp_file = os.path.join(test_dir, f)
                ac_data = read_binary_file(cmp_file, dim=47)
                generated_feats = generate(
                    BATCH_SIZE=args.batch_size, Y_test=ac_data, nice=args.nice)
                out_file = os.path.join(test_dir, fname + '.pls')
                np.savetxt(out_file, generated_feats)
