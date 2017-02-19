import time
import tensorflow as tf
import tensorlayer as tl
from data import load_index, data_to_tfrecord, read_and_decode
from model import inference

learning_rate = 0.1
batch_size = 100
epoches = 100
n_step_epoch = int(20000/100)
n_step = n_step_epoch*epoches
print_freq =1

print("Start.")
with tf.device("/gpu:3"):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    #1.prepare data in cpu
    #print("Prepare Data ...")
    #train_indexs = load_index(is_train=True)
    #test_indexs = load_index(is_train=False)
    #data_to_tfrecord(train_indexs, "train_tfrecord", is_train=True)
    #data_to_tfrecord(test_indexs, "test_tfrecord", is_train=False)
    #print("Data Success.")

    x_train, y_train = read_and_decode("train_tfrecord", is_train=True)
    x_test = read_and_decode("test_tfrecord", is_train=False)

    x_train_batch, hair_batch, hat_batch, \
    gender_batch, top_batch, down_batch, \
    shoes_batch, bag_batch = tf.train.shuffle_batch([x_train, y_train['hair'], 
                                                     y_train['hat'], y_train['gender'],
                                                    y_train['top'], y_train['down'], 
                                                    y_train['shoes'], y_train['bag']],
                                                          batch_size=batch_size,
                                                          capacity=2000,
                                                          min_after_dequeue=1000,
                                                          num_threads=12)
    y_train_batch = {"hat":hat_batch, "hair":hair_batch, "gender":gender_batch,
                  "top":top_batch, "down":down_batch, "shoes":shoes_batch,
                  "bag":bag_batch}
    x_test_train = tf.train.batch([x_test],
                                  batch_size=batch_size,
                                  capacity=2000,
                                  num_threads=12)

    #2.
with tf.device("/gpu:3"):
    cost, acc, network = inference(x_train_batch, y_train_batch, None)

    #cost,
    all_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost['all'])
    hair_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost['hair'])
    hat_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-08, use_locking=False).minimize(cost['hat'])
    gender_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-08, use_locking=False).minimize(cost['gender'])
    top_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-08, use_locking=False).minimize(cost['top'])
    down_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-08, use_locking=False).minimize(cost['down'])
    shoes_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-08, use_locking=False).minimize(cost['shoes'])
    bag_train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-08, use_locking=False).minimize(cost['bag'])
    train_op = {"hair": hair_train_op, "hat": hat_train_op, "gender": gender_train_op,
            "top": top_train_op, "down": down_train_op, "shoes": shoes_train_op, "bag": bag_train_op}
    all_drop = [network["hair"].all_drop, network["hat"].all_drop, network["gender"].all_drop, 
                network["top"].all_drop, network["down"].all_drop, network["shoes"].all_drop,
                network["bag"].all_drop]

with tf.device("/gpu:3"):
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print("Train Start ...")
    step = 0
    for epoch in range(epoches):
        start_time = time.time()
        hat_loss, hair_loss, gender_loss, top_loss, down_loss, shoes_loss, bag_loss = 0, 0, 0, 0, 0, 0, 0
        hat_acc, hair_acc, gender_acc, top_acc, down_acc, shoes_acc, bag_acc = 0, 0, 0, 0, 0, 0, 0
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(n_step_epoch):
            feed_dict = {}
            for i in range(len(all_drop)):
                feed_dict.update(all_drop[i])
            hat_err, hat_ac, _ = sess.run([cost['hat'], acc['hat'], hat_train_op], feed_dict)
            hair_err, hair_ac, _ = sess.run([cost['hair'], acc['hair'], hair_train_op], feed_dict)
            gender_err, gender_ac, _ = sess.run([cost['gender'], acc['gender'], gender_train_op], feed_dict)
            top_err, top_ac, _ = sess.run([cost['top'], acc['top'], top_train_op], feed_dict)
            down_err, down_ac, _ = sess.run([cost['down'], acc['down'], down_train_op], feed_dict)
            shoes_err, shoes_ac, _ = sess.run([cost['shoes'], acc['shoes'], shoes_train_op], feed_dict)
            bag_err, bag_ac, _ = sess.run([cost['bag'], acc['bag'], bag_train_op], feed_dict)
            #err, ac, _ = sess.run([cost['all'], acc['all'], all_train_op],feed_dict)
            step += 1
            
            hat_loss += hat_err
            hair_loss += hair_err
            gender_loss += gender_err
            top_loss += top_err
            down_loss += down_err
            shoes_loss += shoes_err
            bag_loss += bag_err
            
            hat_acc += hat_ac
            hair_acc += hair_ac
            gender_acc += gender_ac
            top_acc += top_ac
            down_acc += down_ac
            shoes_acc += shoes_ac
            bag_acc += bag_ac
            
            train_loss += (hat_err+hair_err+gender_err+top_err+down_err+shoes_err+bag_err)
            train_acc += (hat_ac+hair_ac+gender_ac+top_ac+down_ac+shoes_ac+bag_ac)
            n_batch += 1

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d : Step %d-%d of %d took %fs" % (
            epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            print("     hat loss: %f" % (hat_loss / n_batch))
            print("     hat acc: %f" % (hat_acc / n_batch))
            print("     hair loss: %f" % (hair_loss / n_batch))
            print("     hair acc: %f" % (hair_acc / n_batch))
            print("     gender loss: %f" % (gender_loss / n_batch))
            print("     gender acc: %f" % (gender_acc / n_batch))
            print("     top loss: %f" % (top_loss / n_batch))
            print("     top acc: %f" % (top_acc / n_batch))
            print("     down loss: %f" % (down_loss / n_batch))
            print("     down acc: %f" % (down_acc / n_batch))
            print("     shoes loss: %f" % (shoes_loss / n_batch))
            print("     shoes acc: %f" % (shoes_acc / n_batch))
            print("     bag loss: %f" % (bag_loss / n_batch))
            print("     bag acc: %f" % (bag_acc / n_batch))

    saver = tf.train.Saver()
    save_path = saver.save(sess, "mcnn_model.ckpt")
    coord.request_stop()
    coord.join(threads)
    sess.close()


