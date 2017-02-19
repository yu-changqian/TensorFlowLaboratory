import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep

def conv_lrn_pool(input, conv_shape, conv_strides, pool_size, pool_strides, name):
    with tf.variable_scope("model", None):
        tl.layers.set_name_reuse(None)
        network = tl.layers.Conv2dLayer(input,
                                    act=tf.nn.relu,
                                    shape=conv_shape,
                                    strides=conv_strides,
                                    padding="SAME",
                                    W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                    b_init=tf.constant_initializer(value=0.0),
                                    name="conv_" + name)
        network.outputs = tf.nn.lrn(network.outputs, 5, bias=1.0,
                                alpha=0.00005, beta=0.75, name="norm_" + name)
        network = tl.layers.PoolLayer(network, ksize=pool_size,
                                  strides=pool_strides,
                                  padding="SAME",
                                  pool=tf.nn.max_pool,
                                  name="pool_" + name)
        return network


def branch(input, name):
    with tf.variable_scope("model", None):
        tl.layers.set_name_reuse(None)
        network = tl.layers.DenseLayer(input, n_units=512,
                                   act=tf.nn.relu,
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   b_init=tf.constant_initializer(value=0.0),
                                   name="dense1_" + name)
        network = tl.layers.DropoutLayer(network, keep=0.5, name="drop1_" + name)
        network = tl.layers.DenseLayer(network, n_units=512,
                                   act=tf.nn.relu,
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   b_init=tf.constant_initializer(value=0.0),
                                   name="dense2_" + name)
        network = tl.layers.DropoutLayer(network, keep=0.5, name="drop2_" + name)
        return network


def single_branch(input, name):
    with tf.variable_scope("model", None):
        tl.layers.set_name_reuse(None)
        network = conv_lrn_pool(input, conv_shape=[3, 3, 200, 300],
                            conv_strides=[1, 1, 1, 1],
                            pool_size=[1, 5, 5, 1],
                            pool_strides=[1, 3, 3, 1],name=name)

        network = tl.layers.FlattenLayer(network, name='flatten_'+name)
        network = branch(network, name)
 
        return network


def double_branch(input, name, name1, name2):
    with tf.variable_scope("model", None):
        tl.layers.set_name_reuse(None)
        network = conv_lrn_pool(input, conv_shape=[3, 3, 200, 300],
                            conv_strides=[1, 1, 1, 1],
                            pool_size=[1, 5, 5, 1],
                            pool_strides=[1, 3, 3, 1], name=name)

        network = tl.layers.FlattenLayer(network, name='flatten_'+name)

        branch1 = branch(network, name1)
        branch2 = branch(network, name2)

        return branch1, branch2


def score(input, classes, name):
    with tf.variable_scope("model", None):
        tl.layers.set_name_reuse(None)
        return  tl.layers.DenseLayer(input, n_units=classes,
                                      W_init=tf.truncated_normal_initializer(stddev=0.001),
                                      b_init=tf.constant_initializer(value=0.0),
                                      name=name+"_score")


def loss_acc(y, y_):
     with tf.variable_scope("model", None):
        tl.layers.set_name_reuse(None)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
        correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return cost, acc


def inference(x, y_, reuse):
    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        network = tl.layers.InputLayer(x, name="input_layer")
        network = conv_lrn_pool(network, conv_shape=[7, 7, 3, 75],
                                conv_strides=[1, 2, 2, 1],
                                pool_size=[1, 3, 3, 1],
                                pool_strides=[1, 2, 2, 1], name="layer1")

        network = conv_lrn_pool(network, conv_shape=[5, 5, 75, 200],
                                conv_strides=[1, 2, 2, 1],
                                pool_size=[1, 3, 3, 1],
                                pool_strides=[1, 2, 2, 1], name="layer2")

        hair_output, hat_output = double_branch(network, "head", "hair", "hat")
        hair_score = score(hair_output, 3, "hair")
        hair_y = hair_score.outputs
	if y_ != None:
        	hair_loss, hair_acc = loss_acc(hair_y, y_["hair"])

        hat_score = score(hat_output, 2, "hat")
        hat_y = hat_score.outputs
	if y_ != None:
        	hat_loss, hat_acc = loss_acc(hat_y, y_["hat"])

        gender_output = single_branch(network, "gender")
        gender_score = score(gender_output, 3, "gender")
        gender_y = gender_score.outputs
	if y_ != None:
        	gender_loss, gender_acc = loss_acc(gender_y, y_["gender"])

        top_output = single_branch(network, "top")
        top_score = score(top_output, 6, "top")
        top_y = top_score.outputs
	if y_ != None:
        	top_loss, top_acc = loss_acc(top_y, y_["top"])

        down_output = single_branch(network, "down")
        down_score = score(down_output, 5, "down")
        down_y = down_score.outputs
	if y_ != None:
        	down_loss, down_acc = loss_acc(down_y, y_["down"])

        shoes_output = single_branch(network, "shoes")
        shoes_score = score(shoes_output, 6, "shoes")
        shoes_y = shoes_score.outputs
	if y_ != None:
        	shoes_loss, shoes_acc = loss_acc(shoes_y, y_["shoes"])

        bag_output = single_branch(network, "bag")
        bag_score = score(bag_output, 6, "bag")
        bag_y = bag_score.outputs
	if y_ != None:
        	bag_loss, bag_acc = loss_acc(bag_y, y_["bag"])

	if y_!=None:
		cost = {"all": hair_loss+hat_loss+gender_loss+top_loss+down_loss+shoes_loss+bag_loss,
		        "hair":hair_loss, "hat":hat_loss, "gender":gender_loss,
		        "top":top_loss, "down":down_loss, "shoes":shoes_loss, "bag":bag_loss}
		acc = {"all":hair_acc+hat_acc+gender_acc+top_acc+down_acc+shoes_acc+bag_acc,
		        "hair":hair_acc, "hat":hat_acc, "gender":gender_acc,
		        "top":top_acc, "down":down_acc, "shoes":shoes_acc, "bag":bag_acc}
        net = {"hair":hair_score, "hat":hat_score, "gender":gender_score,
                "top":top_score, "down":down_score, "shoes":shoes_score, "bag":bag_score}

	if y_!=None:
        	return cost, acc, net
	return net


