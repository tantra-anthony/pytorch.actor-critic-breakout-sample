# import libraries as per normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# create functions to initialize the weights
# and set variance of weights
# set a default std for the actor and the critic
def normalized_columns_initializer(weights, std = 1.0):
    # out is a tensor of the no of elemenets of the weights
    out = torch.randn(weights.size())
    # we expand the weights then take the sum then square then square root
    # just like normalizing it
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # var(out) = std^2
    return out

# then create function to initialize the weights of the neural network for optimal learning
def weights_init(m):
    # make distinction between convolution and fully connected layer
    classname = m.__class__.__name__
    # if we have conv connection
    if classname.find('Conv') != -1:
        # get the shape of the weights in the conv connection
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        # w_bound represents the size of the tensor of weights
        w_bound = np.sqrt(6. / fan_in + fan_out)
        # generate random weights inversely proportional to the size of the tensor of weights
        m.weight.data.uniform_(-w_bound, +w_bound) # restrict to lower and upper bound
        m.bias.data.fill_(0)
    # if we have linear conn
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        # full connection has fewer conn than conv
        fan_in = weight_shape[1] # take dim1
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound, +w_bound) # restrict lower and upper bound
        m.bias.data.fill_(0)
        
# create a class for the actor-critic brain
# give long memory to brain using LSTM
# extract features of previous runs
# inherit from nn.Module
class ActorCritic(nn.Module):
    # action_space = no. of possible actions
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__
        # use 32 feature detector using 3 by 3 size feature detectors
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1) # 32 output convoluted images
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        # add LSTM to learn long temporal relationships
        # e.g. if ball hits brick, lstm encodes the bounce
        # so what happens in t+1 depends on t, t-1, and t-2 etc
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # 32 * 3 * 3 is output of conv, can use count_neurons
        num_outputs = action_space.n
        # create full connection for critic
        self.critic_linear = nn.Linear(256, 1) # output = V(S) state
        # for actor, output is all the q-values of the input state, and the action space
        self.actor_linear = nn.Linear(256, num_outputs) # output = Q(S, A)
        self.apply(weights_init)
        # remember small variance for actor but big variance for critic
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        # bias for actor linear
        self.actor_linear.weight.data.fill_(0) # make sure bias is 0
        # critic specifications
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.weight.data.fill_(0) # make sure bias is 0
        # fill LSTM biases to 0 as well
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train() # put model into training mode
        
    # create the forward propagating function
    # not using relu, but elu
    # inputs is not only the input images but also the hidden and cell nodes of lstm
    def forward(self, inputs):
        inputs, (hx, cx) = inputs # convert into a tuple of the hidden, cell and input images
        x = F.elu(self.conv1(inputs)) # insert into first layer, but use non Linear activation
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x)) # propagate through every conv layer
        x = x.view(-1, 32 * 3 * 3) # take one dimen vector
        (hx, cx) = self.lstm(x, (hx, cx)) # spews out two tuples of outputs (for hidden and cell)
        x = hx # hidden nodes are more significant here, so we only extract hx
        # separate actor and critic signals
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        
        
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2)) # kernel size, and stride
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 3, 2)) # use array to make it modular
        # get no of neurons in the flattened layer
        return x.data.view(1, -1).size(1) # take all the pixels of all the channels
    
        
        
        
        