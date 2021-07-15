from neural_network import NeuralNetwork
import gym

def main():

    num_inputs = 24
    num_outputs = 4

    #genotype = [1., 1., 1., 1., 1., 1.]

    #nn = NeuralNetwork(num_inputs, num_outputs, genotype=genotype)
    nn = NeuralNetwork(num_inputs, num_outputs)

    #inputs = [1., 2.]
    #output = nn.forward(inputs)
    #print("Output: ", output)

    nn.print_weights()

    env = gym.make("BipedalWalker-v3")
    print(env.action_space.low)
    print(env.action_space.high)

    state = env.reset()

    while True:

        env.render()

        net_out = nn.forward(state)

        #Normalise output between action space bounds
        action_vals = net_out * (env.action_space.high - env.action_space.low) + \
                      env.action_space.low

        state = env.step(action_vals)[0]

if __name__ == "__main__":
    main()
