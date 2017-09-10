
import numpy as np
import theano
import theano.tensor as tensor
from theano import config

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import dropout, numpy_floatX
from utils import _p
from utils import uniform_weight, zero_bias

from cnn_layer import param_init_encoder, encoder
from lstm_layer import param_init_decoder, decoder

# Set the random number generators' seeds for consistency
SEED = 123  
np.random.seed(SEED)

""" init. parameters. """  
def init_params(options):
    
    n_words = options['n_words']
    n_x = options['n_x']  
    n_h = options['n_h']
    
    params = OrderedDict()
    # word embedding 
    params['Wemb'] = uniform_weight(n_words,n_x)
    #params['Wemb'] = W.astype(config.floatX)
    params['Wemb'][-1] = np.zeros((n_x,)).astype(theano.config.floatX)
    # encoding words into sentences
    length = len(options['filter_shapes'])
    for idx in range(length):
        params = param_init_encoder(options['filter_shapes'][idx],params,prefix=_p('cnn_encoder',idx))
    
    options['n_z'] = options['feature_maps'] * length
    params = param_init_decoder(options,params,prefix='decoder')
    
    params['Vhid'] = uniform_weight(n_h,n_x)
    params['bhid'] = zero_bias(n_words)                                    

    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        #tparams[kk].tag.test_value = params[kk]
    return tparams
    
""" Building model... """

def build_model(tparams,options):
    
    trng = RandomStreams(SEED)
    
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    
    x = tensor.matrix('x', dtype='int32')
    
    layer0_input = tparams['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],tparams['Wemb'].shape[1])) 
    layer0_input = dropout(layer0_input, trng, use_noise)
 
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape, pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input = tensor.concatenate(layer1_inputs,1)
    layer1_input = dropout(layer1_input, trng, use_noise) 

    # description string: n_steps * n_samples
    y = tensor.matrix('y', dtype='int32')
    y_mask = tensor.matrix('y_mask', dtype=config.floatX)  

    n_steps = y.shape[0]
    n_samples = y.shape[1]
    
    n_x = tparams['Wemb'].shape[1]
    
    # n_steps * n_samples * n_x
    y_emb = tparams['Wemb'][y.flatten()].reshape([n_steps,n_samples,n_x])
    y_emb = dropout(y_emb, trng, use_noise)
                                                                                 
    # n_steps * n_samples * n_x
    h_decoder = decoder(tparams, y_emb, layer1_input, mask=y_mask,prefix='decoder')
    h_decoder = dropout(h_decoder, trng, use_noise)
    
    # reconstruct the original sentence
    shape_w = h_decoder.shape
    h_decoder = h_decoder.reshape((shape_w[0]*shape_w[1], shape_w[2]))
    
    # (n_steps * n_samples) * n_words
    Vhid = tensor.dot(tparams['Vhid'],tparams['Wemb'].T)
    pred_w = tensor.dot(h_decoder, Vhid) + tparams['bhid']
    pred_w = tensor.nnet.softmax(pred_w)
    
    x_vec = y.reshape((shape_w[0]*shape_w[1],))
    x_index = tensor.arange(shape_w[0]*shape_w[1])
    x_pred_word = pred_w[x_index, x_vec]
    
    x_mask_reshape = y_mask.reshape((shape_w[0]*shape_w[1],))
    x_index_list = theano.tensor.eq(x_mask_reshape, 1.).nonzero()[0]
    
    x_pred_word_prob = x_pred_word[x_index_list]
    
    cost_x = -tensor.log(x_pred_word_prob + 1e-6).sum()
    
    # the cross-entropy loss 
    num_words = y_mask.sum()                
    cost = cost_x / num_words                            

    return use_noise, x, y, y_mask, cost
