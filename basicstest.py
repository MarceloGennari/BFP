import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[3])
b = tf.placeholder(tf.float32, shape=[3])

c = tf.add(a,b)

with tf.Session() as sess:
	print(sess.run(c, feed_dict={a:[1,2,3], b:[3,4,5]}))
