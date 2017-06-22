import os
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt
from PIL import Image
import time

root = "./ADE20K/images/training/"

def get_filenames(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if ".jpg" in f:
                filenames.append(os.path.join(root, f))
    return filenames

def convert_to_tfrecord():
    writer = tf.python_io.TFRecordWriter("./training.tfrecords")
    filenames = get_filenames(root)
    for name in filenames:
        img = Image.open(name)
        if img.mode == "RGB":
            img = img.resize((256, 256), Image.NEAREST)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                      "img_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()

def read_img(filenames, num_epochs, shuffle=True):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)

    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    img = tf.image.decode_jpeg(value, channels=3)
    img = tf.image.resize_images(img, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return img

def read_tfrecord(filenames, num_epochs, shuffle=True):
    filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs, shuffle=True)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
               "img_raw": tf.FixedLenFeature([], tf.string),
    })
    img = tf.decode_raw(features["img_raw"], tf.uint8)
    img =  tf.reshape(img, [256, 256, 3])

    return img

if __name__ == '__main__':
    #create_tfrecord_start_time = time.time()
    #convert_to_tfrecord()
    #create_tfrecord_duration = time.time() - create_tfrecord_start_time
    #print("Create TFrecord Duration:  %.3f" % (create_tfrecord_duration))

    with tf.Session() as sess:
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3*4

        img = read_img(get_filenames(root), 1, True)
        # img = read_tfrecord("training.tfrecords", 1, True)
        img_batch = tf.train.shuffle_batch([img], batch_size=4, num_threads=8,
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)


        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # print(sess.run(img))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        i = 0
        read_tfrecord_start_time = time.time()
        try:
            while not coord.should_stop():
                imgs = sess.run([img_batch])
                for img in imgs:
                    print(img.shape)
        except Exception, e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        read_tfrecord_duration = time.time() - read_tfrecord_start_time
        print("Read TFrecord Duration:   %.3f" % read_tfrecord_duration)
