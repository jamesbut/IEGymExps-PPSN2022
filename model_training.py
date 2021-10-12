from data import *
from generative_models.gan import *
from generative_models.autoencoder import *
from generative_models.vae import *

def train_gan(train_data_path):

    code_size = 2
    #Set to none if no hidden layer required
    num_hidden_neurons = 64
    training_steps = 2000
    batch_size = 256

    _, _, phenotypes, _, _ = read_data(train_data_path)
    training_data = torch.Tensor(phenotypes)
    #training_data = create_synthetic_data(code_size)

    test = False

    if not test:

        gan = GAN(code_size, training_data, read_generator=False,
                  num_hidden_neurons=num_hidden_neurons)
        gan.train(training_steps, batch_size)
        gan.dump_generator()
        gan.test(plot=True, train_data_dir=train_data_path)
        #gan.test(plot=True)

    else:

        gan = GAN(code_size, training_data, read_generator=True,
                  num_hidden_neurons=num_hidden_neurons)
        gan.test(plot=True, train_data_dir=train_data_path, rand_code=True)


def train_ae(train_data_path):

    code_size = 2
    num_hidden_neurons = 32
    training_steps = 2000
    batch_size=256

    _, _, phenotypes, _, _ = read_data(train_data_path)
    training_data = torch.Tensor(phenotypes)
    #training_data = create_synthetic_data(code_size)

    test=False

    if not test:

        ae = Autoencoder(code_size, training_data, read_decoder=False,
                         num_hidden_neurons=num_hidden_neurons)
        ae.train(training_steps, batch_size)
        ae.dump_decoder()
        ae.test_decoder(plot=True, train_data_dir=train_data_path)

    else:

        ae = Autoencoder(code_size, training_data, read_decoder=True,
                         num_hidden_neurons=num_hidden_neurons)
        ae.test_decoder(plot=True, train_data_dir=train_data_path)


def train_vae(train_data_path):

    code_size = 1
    training_steps = 200
    batch_size = 256

    _, _, phenotypes, _, _ = read_data(train_data_path)
    training_data = torch.Tensor(phenotypes)
    #training_data = create_synthetic_data(code_size)

    vae = VAE(code_size, training_data)

    vae.train(training_steps, batch_size)

    #vae.test()

    vae.dump_decoder()

