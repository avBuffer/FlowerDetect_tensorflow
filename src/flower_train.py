# -*- coding: utf-8 -*-

"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import tensorflow as tf
import numpy as np
import time

from config import cfg
from utils import *
from cnnNet import *


def main(_):            
    print("begin flower train ...")
    start_time = time.time()
    
    w = cfg.width
    h = cfg.height
    c = cfg.channel 
    x_train, y_train, x_val, y_val = get_batch_data(cfg.data_path, w, h, cfg.train_ratio)
    print("reading images cost time: %.2fs" % (time.time() - start_time)) 
    
    # Saving train models
    if not os.path.exists(cfg.model_path):
        os.mkdir(cfg.model_path)
    else :
        del_file(cfg.model_path)
    
    x, y_ = pre_process(w, h, c);
    train_op, loss, acc = build_arch(x, y_, cfg.learn_rate, False)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    batch_size = cfg.batch_size        
    saver=tf.train.Saver()
    sess=tf.Session()  
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(cfg.epochs):
        epoch_time = time.time()
        
        print("epoch: %d" % (epoch))
        #training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err; train_acc += ac; n_batch += 1
        print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
        print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    
        #validating
        if epoch % cfg.val_freq == 0:
            val_loss, val_acc, n_batch = 0, 0, 0
            for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
                err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
                val_loss += err; val_acc += ac; n_batch += 1
            print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
            print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))  
        print("   cost time: %.2fs" % ((time.time() - epoch_time)))    
    
        if epoch % cfg.save_freq == 0:
            saver.save(sess, cfg.model_path + 'flower_model_%04d' % (epoch)) 
            
    sess.close()    
    print("end flower train, cost time: %.2fs" % ((time.time() - start_time))) 

if __name__ == "__main__":
    tf.app.run()
    
    