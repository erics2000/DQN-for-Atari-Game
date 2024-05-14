import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from gameTRY import Breakout
import os
import sys
import time
from train import *
from preprocessing import *

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.01
        self.initial_epsilon = 0.5
        self.number_of_iterations = 1000
        self.epsilon_decay = 0.995
        self.replay_memory_size = 750000
        self.minibatch_size = 32
        self.explore = 3000000

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for the first convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization for the second convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization for the third convolutional layer
        
        # Fully connected layers
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, self.number_of_actions)
        

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Fully connected layers
        x = F.relu(self.fc4(x))
        return self.fc5(x)

def test(model):
    game_state = Breakout()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.take_action(action)
    image_data = preprocessing(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)

        # get action
        action_index = torch.argmax(output)
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.take_action(action)
        image_data_1 = preprocessing(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1

def main(mode):
    if mode == 'test':
        model = torch.load('trained_model/current_model_400000.pth', map_location='cpu').eval()
        test(model)
    elif mode == 'train':
        if not os.path.exists('trained_model/'):
            os.mkdir('trained_model/')
        model = NeuralNetwork()
        model.apply(init_weights)
        start = time.time()
        train(model, start)
    elif mode == 'continue': # You can change trained model id and keep training.
        model = torch.load('trained_model/current_model_100000.pth', map_location='cpu').eval()
        start = time.time()
        train(model, start)

if __name__ == "__main__":
    main(sys.argv[1])


