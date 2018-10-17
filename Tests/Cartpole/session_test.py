import tensorflow as tf

tf.set_random_seed(0)
a=tf.get_variable("a", dtype=tf.float32,initializer=tf.truncated_normal([2,2]))

sess = tf.Session()
sess.__enter__()


sess.run(tf.variables_initializer([a]))
print(sess.run(a))

sess.run(tf.variables_initializer([a]))
print(sess.run(a))

