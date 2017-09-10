'''
Learning Generic Sentence Representations Using Convolutional Neural Networks 
https://arxiv.org/pdf/1611.07897.pdf
Developed by Zhe Gan, zhe.gan@duke.edu, April, 19, 2016
'''

import cPickle
import numpy as np
import theano
import theano.tensor as tensor

from model.autoencoder import init_params, init_tparams
from model.cnn_layer import encoder
from model.utils import get_minibatches_idx
from model.utils import _p

from scipy import spatial


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

def find_sent_embedding(whole, n_words=21102, img_w=300, img_h=48, feature_maps=200, 
    filter_hs=[3,4,5],n_x=300, n_h=600):

    options = {}
    options['n_words'] = n_words
    options['img_w'] = img_w
    options['img_h'] = img_h
    options['feature_maps'] = feature_maps
    options['filter_hs'] = filter_hs
    options['n_x'] = n_x
    options['n_h'] = n_h
    
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
    
    data = np.load('./bookcorpus_result.npz')  
    
    for kk, pp in params.iteritems():
        params[kk] = data[kk]
    
    for kk, pp in params.iteritems():
        tparams[kk].set_value(params[kk])

    x = tensor.matrix('x', dtype='int32')
    
    layer0_input = tparams['Wemb'][tensor.cast(x.flatten(),dtype='int32')].reshape((x.shape[0],1,x.shape[1],tparams['Wemb'].shape[1])) 
 
    layer1_inputs = []
    for i in xrange(len(options['filter_hs'])):
        filter_shape = options['filter_shapes'][i]
        pool_size = options['pool_sizes'][i]
        conv_layer = encoder(tparams, layer0_input,filter_shape, pool_size,prefix=_p('cnn_encoder',i))                          
        layer1_input = conv_layer
        layer1_inputs.append(layer1_input)
    layer1_input = tensor.concatenate(layer1_inputs,1)
                                 
    f_embed = theano.function([x], layer1_input, name='f_embed')
    
    kf = get_minibatches_idx(len(whole), 100)
    sent_emb = np.zeros((len(whole),600))
    
    for i, train_index in kf:
        sents = [whole[t] for t in train_index]
        x = prepare_data_for_cnn(sents)
        sent_emb[train_index[0]:train_index[-1]+1] = f_embed(x)
        if i % 500 == 0:
            print i,
    
    np.savez('./bookcorpus_embedding.npz', sent_emb=sent_emb)
    
    return sent_emb
    
        
def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)
    x.append(0)
    return x
    
def nearest_neightbor(input_sent, z_emb, text_new):
    
    x = get_idx_from_sent(input_sent,wordtoix)
    f_embed = find_sent_embedding(whole, n_words=21102)
    
    x = np.array(x)
    mask = np.ones((len(x)))
    x = x[:,None]
    mask = mask[:,None]
    target_emb = f_embed(x[::-1],mask[::-1])
    
    cos_similarity = []
    for i in range(z_emb.shape[0]):
        vector = z_emb[i]
        result = 1 - spatial.distance.cosine(target_emb, vector)
        cos_similarity.append(result)
    top_indices = np.argsort(cos_similarity)[::-1]
    
    print text_new[top_indices[0]], cos_similarity[top_indices[0]]
    print text_new[top_indices[1]], cos_similarity[top_indices[1]]
    print text_new[top_indices[2]], cos_similarity[top_indices[2]]
    print text_new[top_indices[3]], cos_similarity[top_indices[3]]
    print text_new[top_indices[4]], cos_similarity[top_indices[4]]
    
    return target_emb
    

if __name__ == '__main__':
    
    x = cPickle.load(open("./data/bookcorpus_1M.p","rb"))
    train, val, test = x[0], x[1], x[2]
    train_text, val_text, test_text = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]
    del x
    
    n_words = len(ixtoword)
    
    ixtoword[n_words] = '<pad_zero>'
    wordtoix['<pad_zero>'] = n_words
    n_words = n_words + 1
    
    whole = train + val + test
    whole_text = train_text + val_text + test_text
    del train, val, test
    del train_text, val_text, test_text
    
    sent_emb = find_sent_embedding(whole)
    
    """ sentence retrieval """
    x = np.load('./bookcorpus_embedding.npz')
    sent_emb = x['sent_emb']
    
    x1 = []
    x1.append(get_idx_from_sent("you needed me ?",wordtoix))
    
    idx = 0
    print whole_text[idx]
    target_emb = sent_emb[idx]
    
    cos_similarity = []
    for i in range(len(whole)):
        vector = sent_emb[i]
        result = 1 - spatial.distance.cosine(target_emb, vector)
        cos_similarity.append(result)
    top_indices = np.argsort(cos_similarity)[::-1]
    
    print whole_text[top_indices[0]], cos_similarity[top_indices[0]]
    print whole_text[top_indices[1]], cos_similarity[top_indices[1]]
    print whole_text[top_indices[2]], cos_similarity[top_indices[2]]
    print whole_text[top_indices[3]], cos_similarity[top_indices[3]]
    print whole_text[top_indices[4]], cos_similarity[top_indices[4]]
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
