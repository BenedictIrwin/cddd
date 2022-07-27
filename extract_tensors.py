import tensorflow as tf

init_vars = tf.train.list_variables("./default_model")
arrays = []
names = []
for name , shape in init_vars:
  print(name, shape)
  array = tf.train.load_variable("./default_model", name)
  print(array)
  names.append(name)
  arrays.append(array)
