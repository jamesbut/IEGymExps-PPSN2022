def train_gan(train_data_path):

    training_data = read_data(train_data_path)

    code_size = 1
    training_steps = 20000

    gan = GAN(code_size, training_data)

    gan.train(training_steps)

    gan.test()

    gan.dump_generator()


def train_ae(train_data_path):

    code_size = 1

    training_data = read_data(train_data_path)
    #training_data = create_synthetic_data(code_size)

    training_steps = 20000

    ae = Autoencoder(code_size, training_data)

    ae.train(training_steps)

    #ae.test()

    ae.dump_decoder()

def train_vae(train_data_path):

    code_size = 1

    training_data = read_data(train_data_path)
    #training_data = create_synthetic_data(code_size)

    training_steps = 20000

    vae = VAE(code_size, training_data)

    vae.train(training_steps)

    #vae.test()

    vae.dump_decoder()

