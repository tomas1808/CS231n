import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # lets replace FC with conv layers
    #'''
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size)) #.astype(self.dtype)
    self.params['b1'] = np.zeros((num_filters)) #.astype(self.dtype)

    self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_filters, input_dim[1]/2, input_dim[2]/2)) #.astype(self.dtype)
    self.params['b2'] = np.zeros((hidden_dim)) #.astype(self.dtype)

    self.params['W3'] = np.random.normal(scale=weight_scale, size=(num_classes, hidden_dim, 1, 1)) #.astype(self.dtype)
    self.params['b3'] = np.zeros((num_classes)) #.astype(self.dtype)
    #'''

    '''
    self.params['W1'] = np.zeros((num_filters, input_dim[0], filter_size, filter_size), dtype=self.dtype) #.astype(self.dtype)
    self.params['W1'] += np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size)).astype(self.dtype)
    self.params['b1'] = np.zeros((num_filters), dtype=self.dtype) #.astype(self.dtype)

    self.params['W2'] = np.zeros((hidden_dim, num_filters, input_dim[1]/2, input_dim[2]/2), dtype=self.dtype) #.astype(self.dtype)
    self.params['W2'] += np.random.normal(scale=weight_scale, size=(hidden_dim, num_filters, input_dim[1]/2, input_dim[2]/2)).astype(self.dtype)
    self.params['b2'] = np.zeros((hidden_dim), dtype=self.dtype) #.astype(self.dtype)

    self.params['W3'] = np.zeros((num_classes, hidden_dim, 1, 1), dtype=self.dtype) #.astype(self.dtype)
    self.params['W3'] += np.random.normal(scale=weight_scale, size=(num_classes, hidden_dim, 1, 1)).astype(self.dtype)
    self.params['b3'] = np.zeros((num_classes),dtype=self.dtype) #.astype(self.dtype)
    '''
    # okay lets do it the normal way...
    hidden_input = num_filters * (input_dim[1]/2) * (input_dim[2]/2)

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size)) #.astype(self.dtype)
    self.params['b1'] = np.zeros((num_filters)) #.astype(self.dtype)

    self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_input, hidden_dim))
    self.params['b2'] = np.zeros((hidden_dim)) #.astype(self.dtype)

    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b3'] = np.zeros((num_classes)) #.astype(self.dtype)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    '''
    # conv way
    conv_param2 = {'stride': 1, 'pad': 0}
    conv_param3 = {'stride': 1, 'pad': 0}

    C1, C1cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    C2, C2cache = conv_relu_forward(C1, self.params['W2'], self.params['b2'], conv_param2)
    C3, C3cache = conv_forward_fast(C2, self.params['W3'], self.params['b3'], conv_param3)

    scores = C3 
    '''
    # FC way
    C1, C1cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    C1_flat = np.reshape(C1, (C1.shape[0], (C1.shape[1]*C1.shape[2]*C1.shape[3])))
    L2, L2cache = affine_relu_forward(C1_flat, self.params['W2'], self.params['b2'])
    L3, L3cache = affine_forward(L2, self.params['W3'], self.params['b3'])

    scores = L3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    '''
    loss, dx = softmax_loss(scores, y)
    dx, grads['W3'], grads['b3'] = conv_backward_fast(dx, C3cache)
    dx, grads['W2'], grads['b2'] = conv_relu_backward(dx, C2cache)
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, C1cache)

    loss += 0.5 * self.reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W3'])))
    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2'] 
    grads['W3'] += self.reg * self.params['W3'] 
    '''
    # FC way
    loss, dx = softmax_loss(scores, y)
    dx, grads['W3'], grads['b3'] = affine_backward(dx, L3cache)
    dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, L2cache)
    dx = np.reshape(dx, C1.shape) 
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, C1cache)

    loss += 0.5 * self.reg * (np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) + np.sum(np.square(self.params['W3'])))
    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2'] 
    grads['W3'] += self.reg * self.params['W3'] 
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
class TomConvOnlyNet(object):
  
  '''
  Automatic padding
  Fixed maxpooling to 1/4
  '''
  def __init__(self, input_dim=(3, 32, 32), struct=[['C',3,3,1],['P'],['FC',10]], weight_scale=1e-3, reg=0.0, dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.struct = struct
    self.layer_param = []
    self.bn_params = []

    prev_shape = input_dim

    for i, layer in enumerate(struct):
        if layer[0] == 'C':
            self.params['W' + str(i)] = np.random.normal(scale=weight_scale, size=(layer[1], prev_shape[0], layer[2], layer[2]))
            self.params['b' + str(i)] = np.zeros((layer[1]))
            self.params['gamma' + str(i)] = np.ones((layer[1]))
            self.params['beta' + str(i)] = np.zeros((layer[1]))
            self.bn_params.append({'mode': 'train'})
            self.layer_param.append({'stride': layer[3], 'pad': (layer[2] - 1) / 2})
            prev_shape = (layer[1], prev_shape[1], prev_shape[2])
        elif layer[0] == 'FC':
            self.params['W' + str(i)] = np.random.normal(scale=weight_scale, size=(layer[1], prev_shape[0], prev_shape[1], prev_shape[2]))
            self.params['b' + str(i)] = np.zeros((layer[1]))
            if layer != struct[-1]:
                self.params['gamma' + str(i)] = np.ones((layer[1]))
                self.params['beta' + str(i)] = np.zeros((layer[1]))
                self.bn_params.append({'mode': 'train'})
            self.layer_param.append({'stride': 1, 'pad': 0})
            prev_shape = (layer[1],1,1)
        elif layer[0] == 'P':
            self.bn_params.append({'mode': 'train'})
            self.layer_param.append({'pool_height': 2, 'pool_width': 2, 'stride': 2})
            prev_shape = (prev_shape[0], prev_shape[1]/2, prev_shape[2]/2)
        else:
            raise ValueError('Layer type does not exist')

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    
    # FORWARD PASS #
    scores = X.astype(self.dtype)
    caches = []

    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param[mode] = mode

    for i, layer in enumerate(self.struct):
        if layer[0] == 'C' or layer[0] == 'FC':
            if layer == self.struct[-1]:
                scores, cache = conv_forward_fast(scores, self.params['W'+str(i)], self.params['b'+str(i)], self.layer_param[i])
            else:
                scores, cache = conv_batch_relu_forward(scores, self.params['W'+str(i)], self.params['b'+str(i)], self.params['gamma'+str(i)], self.params['beta'+str(i)], self.layer_param[i], self.bn_params[i])
        elif layer[0] == 'P':
           scores, cache = max_pool_forward_fast(scores, self.layer_param[i])
        caches.append(cache)
    
    scores_shape = scores.shape
    scores = scores[:,:,0,0]
    
    if y is None:
      return scores
    
    # BACKWARD PASS #
    grads = {}
    
    loss, dx = softmax_loss(scores, y)
    dx = dx.reshape(scores_shape) 

    for i, layer in reversed(list(enumerate(self.struct))):
        if layer[0] == 'C' or layer[0] == 'FC':
            if layer == self.struct[-1]:
                dx, grads['W'+str(i)], grads['b'+str(i)] = conv_backward_fast(dx, caches[i]) 
            else:
                dx, grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = conv_batch_relu_backward(dx, caches[i]) 
            loss += 0.5 * self.reg * (np.sum(np.square(self.params['W'+str(i)])))
            grads['W'+str(i)] += self.reg * self.params['W'+str(i)]
        elif layer[0] == 'P':
            dx = max_pool_backward_fast(dx, caches[i])

    return loss, grads
