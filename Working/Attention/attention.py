import tensorflow as tf
import numpy as np

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value
    words_len = inputs.shape[1].value

    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    s_omega = tf.Variable(tf.random_normal([words_len,1], stddev=0.1))


    w = tf.Variable(tf.random_normal([256], stddev=0.1))

    with tf.name_scope('v'):
        v = tf.tanh(tf.tensordot(inputs, w_omega,axes=1) + b_omega)
        v.set_shape([inputs.shape[0].value,inputs.shape[1].value,attention_size])

    vu = tf.tanh(tf.tensordot(v, u_omega, axes=1,name='vu')) 
    vu.set_shape([inputs.shape[0].value,inputs.shape[1].value])
    
    # print(vu.shape)
    # print(s_omega.shape)
    # print("dsf")

    su = tf.matmul(vu, s_omega,name='su') 


    # su.set_shape([inputs.shape[0].value])
    
    print(su.shape)

    batch_size = inputs.shape[0] // 20

    temp = tf.reshape(su,[batch_size,20])

    alphas = tf.nn.softmax(temp, axis=1,name='alphas')


    new_alphas = tf.reshape(alphas,[su.shape[0],1])

    print(alphas.shape)
    output = tf.reduce_sum(inputs * tf.expand_dims(new_alphas, -1), 1)

    print(output.shape)
    if not return_alphas:
        return output
    else:
        return output, alphas
