from data import *
from generative_models.gan import *
from generative_models.autoencoder import *
from generative_models.vae import *

def train_gan(train_data_path):

    code_size = 1
    training_steps = 20000
    batch_size = 256

    #training_data = read_data(train_data_path)
    training_data = create_synthetic_data(code_size)

    gan = GAN(code_size, training_data)

    gan.train(training_steps, batch_size)

    gan.test()

    gan.dump_generator()


def train_ae(train_data_path):

    code_size = 2
    training_steps = 20000
    batch_size=256

    training_data = read_data(train_data_path)
    #training_data = create_synthetic_data(code_size)

    ae = Autoencoder(code_size, training_data)

    ae.train(training_steps, batch_size)

    ae.dump_decoder()

    ae.test()


def train_vae(train_data_path):

    code_size = 1
    training_steps = 20000
    batch_size = 256

    training_data = read_data(train_data_path)
    #training_data = create_synthetic_data(code_size)

    vae = VAE(code_size, training_data)

    vae.train(training_steps, batch_size)

    #vae.test()

    vae.dump_decoder()

