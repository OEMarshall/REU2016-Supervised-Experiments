"""
Deep Belief Net for MINST dataset

DBN uses the code found at: https://github.com/mdenil/dropout with modifications

Modified by Xing Fang
Additional Modifications by Ochaun Marshall, Mutahir Nadeem, Sarbjit Singh

Date: July 20, 2016
"""
import numpy as np
import pickle
import os, sys
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append('/usr/local/lib/python2.7/dist-packages')
#sys.path.remove('/hanaconda/lib/python2.7/site-packages')

import theano
import theano.tensor as T
from theano.ifelse import ifelse
import theano.printing
import theano.tensor.shared_randomstreams
sys.path.append('anaconda/lib/python2.7/site-packages')
#sys.path.append('/home/xing/Documents/DBN/Glorot and Bengio/MINST')

from logistic_sgd import LogisticRegression



##################################
## Various activation functions ##
##################################
#### rectified linear unit
def ReLU(x):
    y = T.nnet.relu(x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)
    

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'
    all_data = pickle.load(open(dataset,"r"))

    #Important: to shuffle the dataset
    rng = np.random.RandomState(111)
    rng.shuffle(all_data)
    
    #pick up the number of training cases
    number_of_training = 150
    
    training_data = [];training_labels = []
    testing_data = []; testing_labels = []
    
    for datapoint in all_data[:number_of_training]:
        training_data.append(datapoint[0])
        training_labels.append(datapoint[1])
    
    for datapoint in all_data[number_of_training:]:
        testing_data.append(datapoint[0])
        testing_labels.append(datapoint[1])
    
    training_data = np.asarray(training_data)
    training_labels = np.asarray(training_labels)
    testing_data = np.asarray(testing_data)
    testing_labels = np.asarray(testing_labels)
    
#    print training_data[:5]
    train_set = zip(training_data,training_labels)
    test_set = zip(testing_data,testing_labels)
    
    train_set = tuple(zip(*train_set))
    test_set = tuple(zip(*test_set))
    
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.
    

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval    
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None, Type = 'Xavier',
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            if Type == 'Xavier':
                W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX)
                W = theano.shared(value=W_values, name='W')
            else:
                W_values = np.asarray(0.01 * rng.standard_normal(
                    size=(n_in, n_out)), dtype=theano.config.floatX)
                W = theano.shared(value=W_values, name='W')
        
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLP(object):
    """A multilayer perceptron with all the trappings required to do dropout
    training.
    """
    def __init__(self,
            rng,
            input,
            layer_sizes,
            dropout_rates,
            activations,
            use_bias=True):

        #rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        #first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rates[0])
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                    input=next_dropout_layer_input,
                    activation=activations[layer_counter],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_counter + 1])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the paramters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                    input=next_layer_input,
                    activation=activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            #first_layer = False
            layer_counter += 1
        
        # Set up the output layer
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
                rng,
                input=next_dropout_layer_input,
                n_in=n_in, n_out=n_out, use_bias=use_bias)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            rng,
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out, use_bias=use_bias)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


def test_mlp(
        initial_learning_rate,
        learning_rate_decay,
        squared_filter_length_limit,
        n_epochs,
        batch_size,
        mom_params,
        activations,
        dropout,
        dropout_rates,
        results_file_name,
        layer_sizes,
        dataset,
        use_bias,
        random_seed,
        decay=True,
        momentum=True,
        L2=True,
        plot = False):
    """
    The dataset is the one from the mlp demo on deeplearning.net.  This training
    function is lifted from there almost exactly.
    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """
    assert len(layer_sizes) - 1 == len(dropout_rates)
    
    # extract the params for momentum
    mom_start = mom_params["start"]
    mom_end = mom_params["end"]
    mom_epoch_interval = mom_params["interval"]
    
    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    epoch = T.scalar()
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    learning_rate = theano.shared(np.asarray(initial_learning_rate,
        dtype=theano.config.floatX))

    rng = np.random.RandomState(random_seed)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x,
                     layer_sizes=layer_sizes,
                     dropout_rates=dropout_rates,
                     activations=activations,
                     use_bias=use_bias)

    # Build the expresson for the cost function.
    if L2:
        lamb = 0.00000001
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)
        if use_bias:
            cost += lamb * sum([(classifier.params[i]**2).sum() for i in range(0,len(classifier.params),2)])/2*batch_size
            dropout_cost += lamb * sum([(classifier.params[i]**2).sum() for i in range(0,len(classifier.params),2)])/2*batch_size
        else:
            cost += lamb *sum([(param**2).sum() for param in classifier.params])/2*batch_size
            dropout_cost += lamb *sum([(param**2).sum() for param in classifier.params])/2*batch_size
    else:
        cost = classifier.negative_log_likelihood(y)
        dropout_cost = classifier.dropout_negative_log_likelihood(y)

    # Compile theano function for testing.
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size:(index + 1) * batch_size],
                y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    #theano.printing.pydotprint(test_model, outfile="test_file.png",
    #        var_with_name_simple=True)

    # Compute gradients of the model wrt parameters
    gparams = []
    for param in classifier.params:
        # Use the right cost function here to train with or without dropout.
        gparam = T.grad(dropout_cost if dropout else cost, param)
        gparams.append(gparam)

    if momentum:
        print >> sys.stderr, ("Using momentum")
        # ... and allocate mmeory for momentum'd versions of the gradient
        gparams_mom = []
        for param in classifier.params:
            gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            gparams_mom.append(gparam_mom)
    
        # Compute momentum for the current epoch
        mom = ifelse(epoch < mom_epoch_interval,
                mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
                mom_end)
    
        
        # Update the step direction using momentum
        updates = OrderedDict()
        for gparam_mom, gparam in zip(gparams_mom, gparams):
            # Misha Denil's original version
            #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
          
            # change the update rule to match Hinton's dropout paper
            updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam
    
        # ... and take a step along that direction
        for param, gparam_mom in zip(classifier.params, gparams_mom):
            # Misha Denil's original version
            #stepped_param = param - learning_rate * updates[gparam_mom]
            
            # since we have included learning_rate in gparam_mom, we don't need it
            # here
            stepped_param = param + updates[gparam_mom]
    
            # This is a silly hack to constrain the norms of the rows of the weight
            # matrices.  This just checks if there are two dimensions to the
            # parameter and constrains it if so... maybe this is a bit silly but it
            # should work for now.
            if param.get_value(borrow=True).ndim == 2:
                #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
                #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
                #updates[param] = stepped_param * scale
                
                # constrain the norms of the COLUMNs of the weight, according to
                # https://github.com/BVLC/caffe/issues/109
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param

    else:
        
        if L2:
            print >> sys.stderr, ("Using gradient decent with L2 regularization")
            updates = [
            (param_i, param_i - learning_rate * (grad_i + lamb*param_i/batch_size))
            for param_i, grad_i in zip(classifier.params, gparams)
            ]
        else:
            print >> sys.stderr, ("Using gradient decent")
            updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(classifier.params, gparams)
            ]
    # Compile theano function for training.  This returns the training cost and
    # updates the model parameters.
    output = dropout_cost if dropout else cost
    train_model = theano.function(inputs=[epoch, index], outputs=output,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]},
            on_unused_input='ignore')
    #theano.printing.pydotprint(train_model, outfile="train_file.png",
    #        var_with_name_simple=True)

    # Theano function to decay the learning rate, this is separate from the
    # training function because we only want to do this once each epoch instead
    # of after each minibatch.
    decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

#    best_params = None
    best_validation_errors = np.inf
    best_test_score = np.inf
    best_iter_valid = 0
    best_iter_test = 0
    test_score = 0.
    epoch_counter = 0
    start_time = time.clock()

#    results_file = open(results_file_name, 'wb')

    if plot:
        plot_training = []
        plot_valid = []
        plot_test = []

    while epoch_counter < n_epochs:
        # Train this epoch
        epoch_counter = epoch_counter + 1
        minibatch_avg_cost = 0
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost += train_model(epoch_counter, minibatch_index)
        
        if plot:
            plot_training.append(minibatch_avg_cost/n_train_batches)
        

#        results_file.write("{0}\n".format(this_validation_errors))
#        results_file.flush()
                
        # test it on the test set
        test_losses = [
            test_model(i)
            for i in xrange(n_test_batches)
        ]
        test_score = np.mean(test_losses)
        
        if plot:
            plot_test.append(test_score)
        
        print(('     epoch %i, test error of '
               'best model %f %%') %
              (epoch_counter, test_score * 100.))
        if test_score < best_test_score:
            best_test_score = test_score
            best_iter_test = epoch_counter
            print >> sys.stderr,('     Current best test score: '+str(best_test_score*100)+'%')
            for i in range(len(classifier.params)):
                pickle.dump(classifier.params[i].get_value(),open('Theta'+str(i)+'.pickle','w'))
                print classifier.params[i].get_value().shape
        
        if decay:
            new_learning_rate = decay_learning_rate()

    end_time = time.clock()
    print >> sys.stderr, (('Best test score of %f %% '
           'obtained at epoch %i') %
          (best_test_score * 100., best_iter_test))      
      
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    if plot:
        Epoch = np.arange(1,n_epochs+1)
        plt.subplot(3, 1, 1)
        plt.plot(Epoch, plot_training)
        plt.grid(axis="y")
        plt.ylabel('Training Error',fontsize=14)
        
        plt.subplot(3, 1, 2)
        plt.plot(Epoch, plot_valid,color="g")
        plt.grid(axis="y")
        plt.xlabel('Iteration',fontsize=18)
        plt.ylabel('Validation Error',fontsize=14)
        
        plt.subplot(3, 1, 3)
        plt.plot(Epoch, plot_test,color="r")
        plt.grid(axis="y")
        plt.xlabel('Iteration',fontsize=18)
        plt.ylabel('Testing Error',fontsize=14)


if __name__ == '__main__':
    
    # set the random seed to enable reproduciable results
    # It is used for initializing the weight matrices
    # and generating the dropout masks for each mini-batch
    random_seed = 23455

    initial_learning_rate = 0.1
    learning_rate_decay = 0.998
    squared_filter_length_limit = 15.0
    n_epochs = 1000
    batch_size = 5
#    layer_sizes = [ 41, 1000, 800, 600, 400, 9 ]
#    layer_sizes = [ 41, 100, 80, 60, 40, 17 ]
    layer_sizes = [ 41, 150, 150, 6 ]
# here    
    
    # activation functions for each layer
    # For this demo, we don't need to set the activation functions for the 
    # on top layer, since it is always 10-way Softmax
#    activations = [ Tanh, Tanh, Tanh, Tanh ]
    activations = [ ReLU, ReLU ]
    
    #### the params for momentum
    mom_start = 0.5
    mom_end = 0.99
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 100
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
                  
    dataset = 'all_data_10000.pickle'

#    if len(sys.argv) < 2:
#        print "Usage: {0} [dropout|backprop]".format(sys.argv[0])
#        exit(1)
#
#    elif sys.argv[1] == "dropout":
#        dropout = True
#        results_file_name = "results_dropout.txt"
#
#    elif sys.argv[1] == "backprop":
#        dropout = False
#        results_file_name = "results_backprop.txt"
#
#    else:
#        print "I don't know how to '{0}'".format(sys.argv[1])
#        exit(1)
    dropout = False
    if dropout:
        # dropout rate for each layer
        dropout_rates = [ 0.2, 0.5, 0.5, 0.5, 0.5] 
    else:
#        dropout_rates = [ 0, 0, 0, 0, 0]
        dropout_rates = [ 0, 0, 0]
    
    use_bias = False
    results_file_name = None

    test_mlp(initial_learning_rate=initial_learning_rate,
             learning_rate_decay=learning_rate_decay,
             squared_filter_length_limit=squared_filter_length_limit,
             n_epochs=n_epochs,
             batch_size=batch_size,
             layer_sizes=layer_sizes,
             mom_params=mom_params,
             activations=activations,
             dropout=dropout,
             dropout_rates=dropout_rates,
             dataset=dataset,
             results_file_name=results_file_name,
             use_bias=use_bias,
             random_seed=random_seed,
             decay=False,
             momentum=False,
             L2=True,
             plot=False)
