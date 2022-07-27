#Adapted from CS231n
from .layers_with_weights import LayerWithWeights
from copy import deepcopy
from abc import abstractmethod
import numpy as np

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

class RNNLayer(LayerWithWeights):
    """ Simple RNN Layer - only calculates hidden states """
    def __init__(self, in_size, out_size):
        """ RNN Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, out_size)
        self.Wh = np.random.rand(out_size, out_size)
        self.b = np.random.rand(out_size)
        self.cache = []
        self.grad = {'dx': None, 'dh0': None, 'dWx': None, 'dWh': None, 'db': None}
        
    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        next_h = np.tanh(x.dot(self.Wx) + prev_h.dot(self.Wh) + self.b) # h(t) funtion. It is output
        cache = (x, prev_h, self.Wx, self.Wh, self.b, next_h)
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        N, T, D = x.shape
        H = h0.shape[1]
        prev_h = h0
        h = np.zeros((N, T, H)) #initiliaze h 
        for i in range(T):
            prev_h, cache_h = self.forward_step(x[:,i,:], prev_h)
            h[:,i,:] = prev_h #hold h(t) values for backward
            self.cache.append(cache_h)
        return h
        
    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
        """
        x, prev_h, Wx, Wh, b, next_h = cache
        dtanh = 1 - next_h**2 #[NxH]
        dnext_tanh = dnext_h * dtanh #[NxH]
        dx = dnext_tanh.dot(Wx.T) #[NxD] #gradient respect to x 
        dprev_h = dnext_tanh.dot(Wh.T) #[NxH] #gradient respect to prev_h
        dWx = (x.T).dot(dnext_tanh) #[DxH] #gradient respect to wx 
        dWh = (prev_h.T).dot(dnext_tanh) #[DxH] #gradient respect to wh
        db = dnext_tanh.sum(axis=0) #[1xH] #gradient respect to b
        return dx, dprev_h, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
            }
        """
        N, T, H = dh.shape
        dxl, dprev_h, dWx, dWh, db = self.backward_step(dh[:,T-1,:], self.cache[T-1])
        D = dxl.shape[1]
        dx = np.zeros((N,T,D)) #initiliaze 
        dx[:,T-1,:] = dxl
        for i in range(T-2, -1, -1):
            dxc, dprev_hc, dWxc, dWhc, dbc = self.backward_step(dh[:,i,:]+dprev_h, self.cache[i]) #calculate gradients 
            dx[:,i,:] += dxc
            dprev_h = dprev_hc
            #update steps 
            dWx += dWxc 
            dWh += dWhc
            db += dbc
        dh0 = dprev_h #first 
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db}
             
class LSTMLayer(LayerWithWeights):
    """ Simple LSTM Layer - only calculates hidden states and cell states """
    def __init__(self, in_size, out_size):
        """ LSTM Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 4 * out_size)
        self.Wh = np.random.rand(out_size, 4 * out_size)
        self.b = np.random.rand(4 * out_size)
        self.cache = []
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None}
        
    def forward_step(self, x, prev_h, prev_c):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
            prev_c: previous cell state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            next_c: next cell state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        H = prev_h.shape[1]
        a = x.dot(self.Wx) + prev_h.dot(self.Wh) + self.b   #[Nx4H]
        a_i = a[:, 0*H:1*H] #[NxH]
        a_f = a[:, 1*H:2*H] #[NxH]
        a_o = a[:, 2*H:3*H] #[NxH]
        a_g = a[:, 3*H:4*H] #[NxH]
    
        #processes in inside cell 
        i_t = sigmoid(a_i) #input gate
        f_t = sigmoid(a_f) #forget gate
        o_t = sigmoid(a_o) #output gate 
        g_t = np.tanh(a_g) #candidate state

        #output gates of cell 
        next_c = f_t * prev_c  + i_t * g_t #cell state 
        next_h = o_t * np.tanh(next_c) #hidden state 
        
        cache = (H, x, self.Wx, self.Wh, a, prev_c, prev_h, next_c, next_h) #cache values

        return next_h, next_c, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Cell state should be initialized to 0.
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        N, T, D = x.shape
        H = h0.shape[1]
        prev_h = h0 #first h 
        prev_c = np.zeros(h0.shape)
        h = np.zeros((N, T, H)) 

        for t in range(T):
            h_t, c_t, cache_t = self.forward_step(x[:,t,:], prev_h, prev_c) #calculate forward values 
            h[:,t,:] = h_t
            prev_h, prev_c = h_t, c_t
            self.cache.append(cache_t) #save for backprop

        return h
        
    def backward_step(self, dnext_h, dnext_c, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            dnext_c: gradient of loss with respect to
                     cell state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dprev_c: gradients of previous cell state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
        """

        H, x, Wx, Wh, a, prev_c, prev_h, next_c, nexht_h  = cache
        a_i, a_f, a_o, a_g = a[:,0*H:1*H], a[:,1*H:2*H], a[:,2*H:3*H], a[:,3*H:4*H] 

        #derivatives of gates according to loss 
        do = dnext_h * np.tanh(next_c) #gradient with respect to output gate
        dc =  dnext_h * sigmoid(a_o) * (1- np.tanh(next_c)**2) #gradient with respect to ct 
        dc += dnext_c #??????????????*
        di = dc * np.tanh(a_g)  #gradient with respect to input gate
        dg = dc * sigmoid(a_i)
        df = dc * prev_c #gradient with respect to forget gate
        dprev_c = dc * sigmoid(a_f) # gradient with respect to ct-1  

        
        da_o = do * (1 - sigmoid(a_o)) * sigmoid(a_o) #Gradient with respect to output gate weights
        da_i = di * (1 - sigmoid(a_i)) * sigmoid(a_i) #Gradient with respect to input gate weights
        da_f = df * (1 - sigmoid(a_f)) * sigmoid(a_f) #Gradient with respect to forget gate weights
        da_g = dg * (1 - np.tanh(a_g) ** 2) #Gradient with respect to inputcell gate weights
        da = np.hstack((da_i, da_f, da_o, da_g))
        
        dx = da.dot(Wx.T) #gradient with respect to x 
        dprev_h = da.dot(Wh.T)  #gradient with respect to ht-1
        db = np.sum(da, axis=0, keepdims=False)  #gradient with respect to db
        dWx = x.T.dot(da)  #gradient with respect to Wx
        dWh = prev_h.T.dot(da)  #gradient with respect to Wh
       
        return dx, dprev_h, dprev_c, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
            }
        """
        N, T, H = dh.shape
        D = self.Wx.shape[0]
        # initialize gradients
        dprev_c = np.zeros((N,H))
        dprev_h = np.zeros((N,H))
        dx = np.zeros((N, T, D))
        dWx = np.zeros((D, 4*H))
        dWh = np.zeros((H, 4*H))
        db = np.zeros(4*H)

        # use of lstm step backward per timestep
        for i in range(T - 1, -1, -1):
            dxi, dprev_h, dprev_c, dWxi, dWhi, dbi = self.backward_step(dprev_h + dh[:,i,:], dprev_c, self.cache[i])
            dx[:,i,:] = dxi
            dWx += dWxi
            dWh += dWhi
            db += dbi

        dh0 = dprev_h
        self.grad = {'dx': dx, 'dh0': dprev_h, 'dWx': dWx, 'dWh': dWh, 'db': db}

class GRULayer(LayerWithWeights):
    """ Simple GRU Layer - only calculates hidden states """
    def __init__(self, in_size, out_size):
        """ GRU Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 2 * out_size)
        self.Wh = np.random.rand(out_size, 2 * out_size)
        self.b = np.random.rand(2 * out_size)
        self.Wxi = np.random.rand(in_size, out_size)
        self.Whi = np.random.rand(out_size, out_size)
        self.bi = np.random.rand(out_size)
        self.cache = []
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None, 'dWxi': None,
                     'dWhi': None, 'dbi': None}
        
    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        H = prev_h.shape[1]
        a = x.dot(self.Wx) + prev_h.dot(self.Wh) + self.b   #[Nx4H]
        a_z = a[:, 0*H:1*H] #[NxH]
        a_r = a[:, 1*H:2*H] #[NxH]
    
    
        #processes in inside cell 
        z_t = sigmoid(a_z) #update gate
        r_t = sigmoid(a_r) #reset gate

        #output gates of cell 
        ai = x.dot(self.Wxi) + (r_t * prev_h).dot(self.Whi) + self.bi  
        h_canditate = np.tanh(ai)
        next_h = z_t * prev_h + (1-z_t) * h_canditate
        
        cache = (H, x, self.Wx, self.Wh, self.b, self.Wxi, self.Whi, self.bi,  a, ai, prev_h, z_t, r_t) #cache values
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        N, T, D = x.shape
        H = h0.shape[1]
        prev_h = h0 #first h 
        h = np.zeros((N, T, H)) 

        for t in range(T):
            h_t, cache_t = self.forward_step(x[:,t,:], prev_h) #calculate forward values 
            h[:,t,:] = h_t
            prev_h = h_t
            self.cache.append(cache_t) #save for backprop

        return h
        
    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWi: gradients of weights Wxi, of shape (D, H)
            dWhi: gradients of weights Whi, of shape (H, H)
            dbi: gradients of bias bi, of shape (H,)
        """
        H, x, Wx, Wh, b, Wxi, Whi, bi,  a, ai, prev_h, z_t, r_t = cache

        dh_canditate = 1- (np.tanh(ai) **2) #Gradient with respect to h_candidate
        dr_t = (dnext_h * (1 - z_t) * dh_canditate).dot(Whi.T) #Gradient with respect to reset gate
        dr_t *= prev_h
        dz_t = dnext_h * (prev_h - np.tanh(ai))  #Gradient with respect to update gate
        dg = dnext_h * (1 - z_t) #gradient with respect to tang
        da_rt = r_t * (1 - r_t) #Gradient with respect to reset gate weights 
        da_zt = z_t * (1 - z_t)  #Gradient with respect to update gate weights

        dz_Wxi = (dnext_h * (1 - z_t) * (1- (np.tanh(ai) **2))).dot(Wxi.T) #Gradient with respect to x 
        dz_Wxz = (da_zt *  dnext_h * (prev_h - np.tanh(ai))).dot(Wx[:,0*H:1*H].T) #Gradient with respect to x-update
        dr_Wxr = (dr_t * da_rt).dot(Wx[:,1*H:2*H].T) #Gradient with respect to x-reset
        dx = dz_Wxi + dr_Wxr  + dz_Wxz #gradients with respect to x 

        #dz_t = dnext_h * (prev_h - np.tanh(ai)) 
        dprev_h =  (dnext_h * z_t) +  ((dz_t * da_zt).dot(Wh[:,0*H:1*H].T))  +  (((1-z_t) * dnext_h * (dh_canditate)).dot(Whi.T) * r_t ) +  ((dr_t  * da_rt).dot(Wh[:,1*H:2*H].T)) #gradients with respect to dprev_h

        da = np.hstack((dz_t * da_zt, dr_t * da_rt)) #stack graiendts
        
        dWx = da.T.dot(x).T
        dWh = da.T.dot(prev_h).T
        db = da.sum(axis=0)
        dWxi = x.T.dot(dg*dh_canditate)
        dWhi = (r_t*prev_h).T.dot(dg*dh_canditate)
        dbi = (dg*dh_canditate).sum(axis=0)

        return dx, dprev_h, dWx, dWh, db, dWxi, dWhi, dbi

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWxi: gradients of weights Wx, of shape (D, H)
            dWhi: gradients of weights Wh, of shape (H, H)
            dbi: gradients of bias b, of shape (H,)
            }
        """
        N, T, H = dh.shape
        D = self.Wx.shape[0]

        # initialize gradients
        dh0 = np.zeros((N, H))
        dprev_h = np.zeros((N,H))
        dx = np.zeros((N, T, D))
        dWx = np.zeros((D, 2*H))
        dWxi = np.zeros((D, H))
        dWh = np.zeros((H, 2*H))
        dWhi = np.zeros((H, H))
        db = np.zeros(2*H)
        dbi = np.zeros(H)

        for i in range(T - 1, -1, -1):
            dx1, dprev_h, dWx1, dWh1, db1, dWxi1, dWhi1, dbi1 = self.backward_step(dprev_h + dh[:,i,:], self.cache[i])
            dx[:,i,:] = dx1
            dWx += dWx1
            dWh += dWh1
            db += db1
            dWxi += dWxi1
            dWhi += dWhi1
            dbi += dbi1

        dh0 = dprev_h

        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db, 'dWxi': dWxi, 'dWhi': dWhi, 'dbi': dbi}

