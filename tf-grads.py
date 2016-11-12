import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.scalar_summary('l2norm/' + name, tf.nn.l2_loss(var))
        tf.scalar_summary('l1norm/' + name, tf.reduce_sum(tf.abs(var)))
        tf.histogram_summary(name, var)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('logdir', 'log', 'Directory to put the logs.')
flags.DEFINE_string('run', '1', 'Directory to put the logs.')

# Create graph
sess = tf.Session()

# Declare batch size
batch_size = 50

x_shape = [3]
# Initialize placeholders
x_data = tf.placeholder(shape=[None, ] + x_shape, dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=x_shape + [1]), name='weights')
b = tf.Variable(tf.random_normal(shape=[1, 1]), name='bias')

# Declare model operations
model_output = tf.add(tf.matmul(x_data, A), b, name='activation')
variable_summaries(model_output, model_output.name)

# Declare the elastic net loss function
elastic_param1 = tf.constant(1.)
elastic_param2 = tf.constant(1.)
l1_a_loss = tf.reduce_mean(tf.abs(A))
l2_a_loss = tf.reduce_mean(tf.square(A))
e1_term = tf.mul(elastic_param1, l1_a_loss, name='l1_reg')
e2_term = tf.mul(elastic_param2, l2_a_loss, name='l2_reg')
ce_loss = tf.reduce_mean(tf.square(model_output - y_target), name='ce_loss')
total_loss = ce_loss + e1_term + e2_term

tf.scalar_summary('l1 loss', l1_a_loss)
tf.scalar_summary('l2 loss', l2_a_loss)
tf.scalar_summary('ce loss', ce_loss)
tf.scalar_summary('total loss', total_loss)

# Initialize variables
init = tf.initialize_all_variables()
sess.run(init)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.001)

grads_vars = my_opt.compute_gradients(loss=ce_loss)
for grad, trainable_var in grads_vars:
    variable_summaries(var=grad, name=grad.name)
    variable_summaries(var=trainable_var, name=trainable_var.name)
train_step = my_opt.apply_gradients(grads_and_vars=grads_vars)

# Load the data
iris = datasets.load_iris()
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
x_vals = np.array([[x[1], x[2], x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

merged = tf.merge_all_summaries()

train_writer = tf.train.SummaryWriter(FLAGS.logdir + '/train/' + FLAGS.run,
                                      sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.logdir + '/test/' + FLAGS.run)

# Training loop
loss_vec = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    summary, temp_loss, _ = sess.run([merged, total_loss, train_step], feed_dict={x_data: rand_x, y_target: rand_y})
    train_writer.add_summary(summary=summary, global_step=i)

    print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
    print('Loss = ' + str(temp_loss))

# Get the optimal coefficients
[[sw_coef], [pl_coef], [pw_ceof]] = sess.run(A)
[y_intercept] = sess.run(b)
