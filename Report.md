# Report

In this repository we provide a solution to the Banana environment using deep Q learning (DQN), double deep Q learning (DDQN) and the use of a dueling network. 

## Learning algorithm

Our reinforcement learning algorithm needs at least 4 things to function properly. 

1. The environment and a connection to python (provided by Banana.exe and unity-agents)
2. The our jupyter notebook that plays episodes, collects rewards and supplies information to our reinforcement learning algorithm.
3. Our deep neural network that estimates the value of an action at a parcticular state.
4. Our DQN (or other) learning algorithm.


We will describe our learning algorithm and model in the next section

### Deep neural network

In previous exercises we have always kept a Q-table with Q-values for every state in the table. This was easy to perform when our statespace was very low and discrete. By using a neural network we can instead use an approximation of our Q-values instead of the actual Q-value. Using a neural network can drastically reduce the amount of paramaters that need to be stored in memory and can handle continous values.

For our neural network we have chosen 4 layers of 48 units each. Each layer except the final layer uses the Relu non-linearity function to apply non-linearity to our network. Our final layer has 4 output units, each estimating the Q-value for each of the 4 actions.

I have chosen an input layer of 48 units so that each state can potentially be a direct mapping of a single dimension of the state. I feel that this does not squeeze our dimensions down and does not lose any information in the input layer. I have chosen a deeper network, which experimentally seemed to work better. The amount of units the same throughout the network is kept identical, because there does not seem to be a benefit to reducing the layer size and computationally would not yield a speedup anyway.

#### DQN Learning algorithm

The learning algorithm that is used consists of 3 things.
1. Update the network.
1. Experience replay.
2. Fixed Q-targets.