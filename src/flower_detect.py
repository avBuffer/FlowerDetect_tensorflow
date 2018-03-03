# -*- coding: utf-8 -*-

"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

from skimage import io,transform
import tensorflow as tf
import numpy as np
import time

from config import cfg
from utils import *


def main(_):      
    print("begin flower detect ...")
    start_time = time.time()
      
    flower_labels = get_labels(cfg.data_path)
    #print('flower labels=%s' % (flower_labels))
           
    test_path = cfg.test_path
    image_names = sorted(os.listdir(test_path))            
    images = []
    idx = 0
    for image_name in image_names:
        #print("idx: %d, image_name: %s" % (idx, test_path + image_name))
        image = read_one_image(test_path + image_name, cfg.width, cfg.height)
        images.append(image)
        idx = idx + 1
  
    cost_time = time.time()-start_time    
    print("reading images cost time: total=%.2fs average=%.4fs" % (cost_time, cost_time/idx))

    # Saving detect results
    result_file = cfg.result_path + 'flower_result.txt'
    if not os.path.exists(cfg.result_path):
        os.mkdir(cfg.result_path)
    else :
        del_file(cfg.result_path)
    
    fd_results = open(result_file, 'w')
    fd_results.write('index,image,type,score\n')     

    with tf.Session() as sess:     
        saver = tf.train.import_meta_graph('../checkpoints/flower_model_0500.meta')
        saver.restore(sess,tf.train.latest_checkpoint('../checkpoints/'))
    
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x:images}
    
        logits = graph.get_tensor_by_name("logits_eval:0")    
        classification_result = sess.run(logits, feed_dict)
    
        # print prediction matrix
        #print(classification_result)
        
        # print maximum value of each row of the prediction matrix
        #print(tf.argmax(classification_result,1).eval())
        
        # classify flower type according to index in flower labels
        output = tf.argmax(classification_result,1).eval()
        for i in range(len(output)):            
            print("No: %d, flower: %s is %s, score: %.2f" % (i, image_names[i], flower_labels[output[i]], max(classification_result[i])))            
            pil_draw_image(cfg.test_path, cfg.result_path, image_names[i], flower_labels[output[i]])
            
            fd_results.write(str(i) + ',' + image_names[i] + ',' + flower_labels[output[i]] + ',' + str(max(classification_result[i])) + '\n')
            fd_results.flush()             
    
    fd_results.close()          
    cost_time = time.time()-start_time
    print("end flower detect cost time: total=%.2fs average=%.4fs" % (cost_time, cost_time/idx))
      
if __name__ == "__main__":
    tf.app.run()
       