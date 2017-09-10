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
from model.utils import _p

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

def find_sent_embedding(n_words=21102, img_w=300, img_h=48, feature_maps=200, 
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
    
    return f_embed, params
       
def predict(z, params, beam_size, max_step, prefix='decoder'):
    
    """ z: size of (n_z, 1)
    """
    n_h = params[_p(prefix,'U')].shape[0]
    
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1)*dim]
        
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    Vhid = np.dot(params['Vhid'],params['Wemb'].T)
    
    def _step(x_prev, h_prev, c_prev):
        preact = np.dot(h_prev, params[_p(prefix, 'U')]) + \
            np.dot(x_prev, params[_p(prefix, 'W')]) + \
            np.dot(z, params[_p(prefix, 'C')]) + params[_p(prefix, 'b')]
        
        i = sigmoid(_slice(preact, 0, n_h))
        f = sigmoid(_slice(preact, 1, n_h))
        o = sigmoid(_slice(preact, 2, n_h))
        c = np.tanh(_slice(preact, 3, n_h))
        
        c = f * c_prev + i * c
        h = o * np.tanh(c)
        
        y = np.dot(h, Vhid) + params['bhid']  
        
        return y, h, c
        
    h0 = np.tanh(np.dot(z, params[_p(prefix, 'C0')]) + params[_p(prefix, 'b0')])
    y0 = np.dot(h0, Vhid) + params['bhid']  
    c0 = np.zeros(h0.shape)
    
    maxy0 = np.amax(y0)
    e0 = np.exp(y0 - maxy0) # for numerical stability shift into good numerical range
    p0 = e0 / np.sum(e0)
    y0 = np.log(1e-20 + p0) # and back to log domain
    
    beams = []
    nsteps = 1
    # generate the first word
    top_indices = np.argsort(-y0)  # we do -y because we want decreasing order
    
    for i in xrange(beam_size):
        wordix = top_indices[i]
        # log probability, indices of words predicted in this beam so far, and the hidden and cell states
        beams.append((y0[wordix], [wordix], h0, c0))
    
    # perform BEAM search. 
    if beam_size > 1:
        # generate the rest n words
        while True:
            beam_candidates = []
            for b in beams:
                ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
                if ixprev == 0 and b[1]:
                    # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue
                (y1, h1, c1) = _step(params['Wemb'][ixprev], b[2], b[3])
                y1 = y1.ravel() # make into 1D vector
                maxy1 = np.amax(y1)
                e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
                p1 = e1 / np.sum(e1)
                y1 = np.log(1e-20 + p1) # and back to log domain
                top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
                for i in xrange(beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
            beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size] # truncate to get new beams
            nsteps += 1
            if nsteps >= max_step: # bad things are probably happening, break out
                break
        # strip the intermediates
        predictions = [(b[0], b[1]) for b in beams]
    else:
        nsteps = 1
        h = h0
        # generate the first word
        top_indices = np.argsort(-y0)  # we do -y because we want decreasing order
        ixprev = top_indices[0]
        predix = [ixprev]
        predlogprob = y0[ixprev]
        while True:
            (y1, h) = _step(params['Wemb'][ixprev], h)
            ixprev, ixlogprob = ymax(y1)
            predix.append(ixprev)
            predlogprob += ixlogprob
            nsteps += 1
            if nsteps >= max_step:
                break
            predictions = [(predlogprob, predix)]
        
    return predictions
    
def ymax(y):
    """ simple helper function here that takes unnormalized logprobs """
    y1 = y.ravel() # make sure 1d
    maxy1 = np.amax(y1)
    e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
    p1 = e1 / np.sum(e1)
    y1 = np.log(1e-20 + p1) # guard against zero probabilities just in case
    ix = np.argmax(y1)
    return (ix, y1[ix]) 

def generate(z_emb, params):
    
    predset = []
    for i in xrange(len(z_emb)):
        pred = predict(z_emb[i], params, beam_size=5, max_step=40)
        predset.append(pred)
        #print i,

    return predset 

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
    

if __name__ == '__main__':
    
    print "loading data..."
    x = cPickle.load(open("./data/bookcorpus_1M.p","rb"))
    train, val, test = x[0], x[1], x[2]
    train_text, val_text, test_text = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]
    del x
    
    n_words = len(ixtoword)
    ixtoword[n_words] = '<pad_zero>'
    wordtoix['<pad_zero>'] = n_words
    n_words = n_words + 1
    
    f_embed, params = find_sent_embedding()
    
    x1 = []
    x1.append(get_idx_from_sent("you needed me ?",wordtoix))
    x1.append(get_idx_from_sent("you got me ?",wordtoix))
    x1.append(get_idx_from_sent("i got you .",wordtoix))
    
    x2 = []
    x2.append(get_idx_from_sent("this is great .",wordtoix))
    x2.append(get_idx_from_sent("this is awesome .",wordtoix))
    x2.append(get_idx_from_sent("you are awesome .",wordtoix))
    
    x3 = []
    x3.append(get_idx_from_sent("its lovely to see you .",wordtoix))
    x3.append(get_idx_from_sent("its great to meet you .",wordtoix))
    x3.append(get_idx_from_sent("its great to meet him .",wordtoix))
    
    x4 = []
    x4.append(get_idx_from_sent("he had thought he was going crazy .",wordtoix))
    x4.append(get_idx_from_sent("i felt like i was going crazy .",wordtoix))
    x4.append(get_idx_from_sent("i felt like to say the right thing .",wordtoix))
    
    sent_emb = f_embed(prepare_data_for_cnn(x1))
    sent_emb_x1 = sent_emb[0] - sent_emb[1] + sent_emb[2]
    
    sent_emb = f_embed(prepare_data_for_cnn(x2))
    sent_emb_x2 = sent_emb[0] - sent_emb[1] + sent_emb[2]
    
    sent_emb = f_embed(prepare_data_for_cnn(x3))
    sent_emb_x3 = sent_emb[0] - sent_emb[1] + sent_emb[2]
    
    sent_emb = f_embed(prepare_data_for_cnn(x4))
    sent_emb_x4 = sent_emb[0] - sent_emb[1] + sent_emb[2]
    
    sent_emb = np.stack((sent_emb_x1,sent_emb_x2,sent_emb_x3,sent_emb_x4))
    
    predset = generate(sent_emb, params)
        
    predset_text = []
    for sent in predset:
        rev = []
        for sen in sent:
            smal = []
            for w in sen[1]:
                smal.append(ixtoword[w])
            rev.append(' '.join(smal))
        predset_text.append(rev)
    
    for i in range(4):
        print predset_text[i][0]
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
