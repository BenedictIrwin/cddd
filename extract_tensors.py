import tensorflow as tf
import numpy as np

init_vars = tf.train.list_variables("./default_model")
arrays = []
names = []
for name , shape in init_vars:
  if("optimizer" in name):
    continue
  print(name, shape)
  save_name = name.replace("/","-")
  array = tf.train.load_variable("./default_model", name)
  print(array)
  #names.append(name)
  #arrays.append(array)
  np.save(save_name, array)


