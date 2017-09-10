'''
Learning Generic Sentence Representations Using Convolutional Neural Networks 
https://arxiv.org/pdf/1611.07897.pdf
Developed by Zhe Gan, zhe.gan@duke.edu, April, 19, 2016
'''

#import os
import time
import logging
import cPickle

import numpy as np
import theano
import theano.tensor as tensor

from model.autoencoder import init_params, init_tparams, build_model
from model.optimizers import Adam
from model.utils import get_minibatches_idx, unzip

#theano.config.optimizer='fast_compile'
#theano.config.exception_verbosity='high'
#theano.config.compute_test_value = 'warn'

def prepare_data_for_cnn(seqs_x, maxlen=40, n_words=21103, filter_h=5):
    
    lengths_x = [len(s) for s in seqs_x]
    
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            # we need pad the special <pad_zero> token.
            xx.append(n_words-1)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(n_words-1)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x

def prepare_data_for_rnn(seqs_x, maxlen=40):
    
    lengths_x = [len(s) for s in seqs_x]
    
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)

    x = np.zeros((maxlen_x, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen_x, n_samples)).astype(theano.config.floatX)
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx], idx] = 1.

    return x, x_mask
    
def calu_cost(f_cost, prepare_data_for_cnn, prepare_data_for_rnn, data, kf):

    total_negll = 0.
    total_len = 0.
    
    for _, train_index in kf:
        sents = [train[t]for t in train_index]
                
        x = prepare_data_for_cnn(sents)
        y, y_mask = prepare_data_for_rnn(sents)
        negll = f_cost(x, y, y_mask) * np.sum(y_mask)
        length = np.sum(y_mask)
        total_negll += negll
        total_len += length

    return total_negll/total_len

""" Training the model. """

def train_model(train, val, test, n_words=21103, img_w=300, max_len=40, 
    feature_maps=200, filter_hs=[3,4,5], n_x=300, n_h=600, 
    max_epochs=8, lrate=0.0002, batch_size=64, valid_batch_size=64, dispFreq=10, 
    validFreq=500, saveFreq=1000, saveto = 'bookcorpus_result.npz'):
        
    """ train, valid, test : datasets
        n_words : vocabulary size
        img_w : word embedding dimension, must be 300.
        max_len : the maximum length of a sentence 
        feature_maps : the number of feature maps we used 
        filter_hs: the filter window sizes we used
        n_x: word embedding dimension
        n_h: the number of hidden units in LSTM        
        max_epochs : the maximum number of epoch to run
        lrate : learning rate
        batch_size : batch size during training
        valid_batch_size : The batch size used for validation/test set
        dispFreq : Display to stdout the training progress every N updates
        validFreq : Compute the validation error after this number of update.
        saveFreq: save the result after this number of update.
        saveto: where to save the result.
    """
    
    img_h = max_len + 2*(filter_hs[-1]-1)
    
    options = {}
    options['n_words'] = n_words
    options['img_w'] = img_w
    options['img_h'] = img_h
    options['feature_maps'] = feature_maps
    options['filter_hs'] = filter_hs
    options['n_x'] = n_x
    options['n_h'] = n_h
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['saveFreq'] = saveFreq
   
    logger.info('Model options {}'.format(options))

    logger.info('Building model...')
    
    filter_w = img_w
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
        
    options['filter_shapes'] = filter_shapes
    options['pool_sizes'] = pool_sizes
    
    params = init_params(options)
    tparams = init_tparams(params)

    use_noise, x, y, y_mask, cost = build_model(tparams,options)
    
    f_cost = theano.function([x, y, y_mask], cost, name='f_cost')
    
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = Adam(tparams, cost, [x, y, y_mask], lr)
    
    logger.info('Training model...')
    
    history_cost = []  
    uidx = 0  # the number of update done
    start_time = time.time()
    
    kf_valid = get_minibatches_idx(len(val), valid_batch_size)
    
    zero_vec_tensor = tensor.vector()
    zero_vec = np.zeros(img_w).astype(theano.config.floatX)
    set_zero = theano.function([zero_vec_tensor], updates=[(tparams['Wemb'], tensor.set_subtensor(tparams['Wemb'][21102,:], zero_vec_tensor))])
    
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
            
            kf = get_minibatches_idx(len(train), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(0.)

                sents = [train[t]for t in train_index]
                
                x = prepare_data_for_cnn(sents)
                y, y_mask = prepare_data_for_rnn(sents)
                n_samples += y.shape[1]

                cost = f_grad_shared(x, y, y_mask)
                f_update(lrate)
                # the special <pad_zero> token does not need to update.
                set_zero(zero_vec)

                if np.isnan(cost) or np.isinf(cost):
                    
                    logger.info('NaN detected')
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    logger.info('Epoch {} Update {} Cost {}'.format(eidx, uidx, np.exp(cost)))
                
                if np.mod(uidx, saveFreq) == 0:
                    
                    logger.info('Saving ...')
                    
                    params = unzip(tparams)
                    np.savez(saveto, history_cost=history_cost, **params)
                    
                    logger.info('Done ...')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    
                    valid_cost = calu_cost(f_cost, prepare_data_for_cnn, prepare_data_for_rnn, val, kf_valid)
                    history_cost.append([valid_cost])
                        
                    logger.info('Valid {}'.format(np.exp(valid_cost)))

        logger.info('Seen {} samples'.format(n_samples))

    except KeyboardInterrupt:
        logger.info('Training interupted')

    end_time = time.time()
    
#    if best_p is not None:
#        zipp(best_p, tparams)
#    else:
#        best_p = unzip(tparams)
    
    
    use_noise.set_value(0.)
    valid_cost = calu_cost(f_cost, prepare_data_for_cnn, prepare_data_for_rnn, val, kf_valid)
    logger.info('Valid {}'.format(np.exp(valid_cost)))
    
    params = unzip(tparams)
    np.savez(saveto, history_cost=history_cost, **params)

    
    logger.info('The code run for {} epochs, with {} sec/epochs'.format(eidx + 1, 
                 (end_time - start_time) / (1. * (eidx + 1))))
    
    
    return valid_cost

if __name__ == '__main__':
    
    logger = logging.getLogger('train_autoencoder')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('train_autoencoder.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    
    x = cPickle.load(open("./data/bookcorpus_1M.p","rb"))
    train, val, test = x[0], x[1], x[2]
    train_text, val_text, test_text = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]
    del x
    del train_text, val_text, test_text
    
    n_words = len(ixtoword)
    ixtoword[n_words] = '<pad_zero>'
    wordtoix['<pad_zero>'] = n_words
    n_words = n_words + 1
    
    valid_cost = train_model(train, val, test, n_words=n_words)
