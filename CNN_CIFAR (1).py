
# coding: utf-8

# In[53]:


import tensorflow as tf
import numpy as np


print('hello')


# In[54]:


# Put file path as a string here
CIFAR_DIR = '/Users/ajay/Downloads/Python_ML/CIFAR-10/cifar-10-batches-py/'


# In[55]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict



# In[56]:


def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


# In[57]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']


# In[58]:


all_data = [0,1,2,3,4,5,6]


# In[59]:


def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


# In[60]:


for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

    
batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]


# In[61]:


type(all_data[1])
batch_meta


data_batch1.keys()

data_batch1[b'data']




X = data_batch1[b'data']
X_filter = X.reshape(10000,3,32,32)
Y = X_filter.transpose
Y1 = X_filter.transpose(0,2,3,1)

Y1.shape



import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import numpy as np




def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


# In[62]:


test_data = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
test_data1 = np.vstack([d[b'data'] for d in test_data ])
test_data1.shape
test_length = len(test_data1)
test_data1 = test_data1.reshape(test_length,3,32,32).transpose(0,2,3,1)
test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in test_data]), 10)


# In[63]:


Z = test_data1
M = test_labels
Z.shape


# In[64]:


LR = 0.0001
n = 32*32*3
x = tf.placeholder(tf.float32,shape=[None,n])
y = tf.placeholder(tf.float32,shape=[None,10])
x = tf.reshape(x,[-1,32,32,3])
#y_true = tf.reshape(y_true,[-1,10])
hold_dropout = tf.placeholder(tf.float32, shape= None)
x


# In[65]:


def bias_var(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[66]:


def weight_init(x):
    init_random_dist = tf.truncated_normal(x, stddev=0.1)
    return tf.Variable(init_random_dist)


# In[68]:


def convolution(p):
    #Convo1
    W1 = weight_init([1,1,3,100])
    S1 = [1,1,1,1]
    b1 = bias_var([100])
    conv = tf.add(tf.nn.conv2d(p,W1,S1, padding = 'SAME'),b1)
    conv_x = tf.nn.relu(conv)
    z = tf.nn.max_pool(conv_x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #Convo2
    W2 = weight_init([2,2,100,20])
    b2 = bias_var([20])
    conv2 = tf.add(tf.nn.conv2d(z,W2,S1,padding='SAME'),b2)
    conv_x2 = tf.nn.relu(conv2)
    z2 = tf.nn.max_pool(conv_x2,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
    #flatten
    z2 = tf.reshape(z2,[-1,8*8*20])
    #1rst Connected Layer 
    input_size = int(z2.get_shape()[1])
    n_size = 10
    W3 = weight_init([input_size,n_size])
    b3 =bias_var([10])
    
    c  = tf.add(tf.matmul(z2,W3),b3) 
    
    return c


# In[69]:


def Conv_nueral1(p):
    #Convo1
    W1 = weight_init([1,1,3,100])
    S1 = [1,1,1,1]
    b1 = bias_var([100])
    conv = tf.add(tf.nn.conv2d(p,W1,S1, padding = 'SAME'),b1)
    conv_x = tf.nn.relu(conv)
    z = tf.nn.max_pool(conv_x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    #Convo2
    W2 = weight_init([2,2,100,20])
    b2 = bias_var([20])
    conv2 = tf.add(tf.nn.conv2d(z,W2,S1,padding='SAME'),b2)
    conv_x2 = tf.nn.relu(conv2)
    z2 = tf.nn.max_pool(conv_x2,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')
    #flatten
    z2 = tf.reshape(z2,[-1,8*8*20])
    #1rst Connected Layer 
    input_size = int(z2.get_shape()[1])
    n_size = 10
    W3 = weight_init([input_size,n_size])
    b3 =bias_var([10])
    return tf.add(tf.matmul(z2,W3),b3)  


# In[71]:


pred =convolution(x)


# In[72]:


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =y, logits = pred))


# In[73]:


optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(cost)


# In[ ]:


# Launch the session
sess = tf.InteractiveSession()
# Intialize all the variables
init = tf.initialize_all_variables()
sess.run(init)
# Training Epochs
# Essentially the max amount of loops possible before we stop
# May stop earlier if cost/loss limit was set
n_samples = 5000
batch_size = 100
for epoch in range(8):

    # Start with cost = 0.0
    avg = 0.0
    

    # Convert total number of batches to integer
    total_batch = int(n_samples/batch_size)

    # Loop over all batches
    for i in range(total_batch):

        # Grab the next batch of training data and labels
        batch_x, batch_y = Z,M

        # Feed dictionary for optimization and loss value
        # Returns a tuple, but we only need 'c' the cost
        # So we set an underscore as a "throwaway"
        a, c = sess.run([optimizer, cost], feed_dict=({x:batch_x,y:batch_y}))
        
        avg +=c/total_batch
        
        
    print("Epoch: {} cost={:.4f}".format(epoch+1,avg))
    


