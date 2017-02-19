import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from xml.etree.ElementTree import ElementTree as ET

img_height = 256
img_width = 128


def load_index(is_train):
    if is_train == True:
        index_file = "../TRAIN/indexs.txt"
    else:
        index_file = "../TEST/indexs.txt"

    with open(index_file) as f:
        indexs = f.readlines()

    indexs = [i.strip() for i in indexs]

    return indexs


def _load_img(idx, is_train):
    if is_train == True:
        img_path = "../TRAIN/IMAGES_TRAIN"
    else:
        img_path = "../TEST/IMAGES_TEST"

    im = Image.open("{}/{}.jpg".format(img_path, idx))
    im = im.resize((img_width, img_height))
    in_ = np.array(im, dtype=np.uint8)

    #in_ = in_.transpose((2, 0, 1))
    return in_


def _load_label(idx):
    xml_path = "../TRAIN/ANNOTATIONS_TRAIN"

    tree = ET()
    tree.parse("{}/{}.xml".format(xml_path, idx))
    labels = {}
    labels["hair"] = int(tree.find("hairlength").text)
    labels["gender"] = int(tree.find("gender").text)

    objs = tree.findall("subcomponent")
    for obj in objs:
        name = obj.find("name").text
        if name in ["top", "down", "shoes", "bag"]:
            category = obj.find("category").text
            if category == "NULL" and name in ["bag", "shoes"]:
                category = 5  # represent there is no bag or shoes
            labels[name] = int(category)
        if name == "hat":
            if obj.find("bndbox").find("xmin").text == "NULL":
                labels[name] = 0 # represent here is no hat
            else:
                labels[name] = 1
    return labels


def data_to_tfrecord(indexs, filename, is_train):
    print("Converting data into %s ..."%filename)
    writer = tf.python_io.TFRecordWriter(filename)

    for idx in indexs:
        img = _load_img(idx, is_train)
	#tl.visualize.frame(I=img, second=5, saveable=False, name='frame', fig_idx=12836)
        img_raw = img.tobytes()
        if is_train == True:
            labels = _load_label(idx)
            hat_label = int(labels['hat'])
            hair_label = int(labels['hair'])
            gender_label = int(labels['gender'])
            top_label = int(labels['top'])
            down_label = int(labels['down'])
            shoes_label = int(labels['shoes'])
            bag_label = int(labels['bag'])

            example = tf.train.Example(features=tf.train.Features(feature={
                "hat_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[hat_label])),
                "hair_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[hair_label])),
                "gender_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[gender_label])),
                "top_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[top_label])),
                "down_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[down_label])),
                "shoes_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[shoes_label])),
                "bag_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[bag_label])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()



def read_and_decode(filename, is_train):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    if is_train == True:
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               "hat_label": tf.FixedLenFeature([], tf.int64),
                                               "hair_label": tf.FixedLenFeature([], tf.int64),
                                               "gender_label": tf.FixedLenFeature([], tf.int64),
                                               "top_label": tf.FixedLenFeature([], tf.int64),
                                               "down_label": tf.FixedLenFeature([], tf.int64),
                                               "shoes_label": tf.FixedLenFeature([], tf.int64),
                                               "bag_label": tf.FixedLenFeature([], tf.int64),
                                               "img_raw": tf.FixedLenFeature([], tf.string),
                                           })
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [128, 256, 3])
	#image = Image.frombytes('RGB', (224, 224), img[0])
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	#print(type(img))
	#img = np.asarray(img, dtype=np.uint8)
	#print(type(img))
	#tl.visualize.frame(I=img, second=5, saveable=False, name='frame', fig_idx=12836)

        hat_label = tf.cast(features['hat_label'], tf.int32)
        hair_label = tf.cast(features['hair_label'], tf.int32)
        gender_label = tf.cast(features['gender_label'], tf.int32)
        top_label = tf.cast(features['top_label'], tf.int32)
        down_label = tf.cast(features['down_label'], tf.int32)
        shoes_label = tf.cast(features['shoes_label'], tf.int32)
        bag_label = tf.cast(features['bag_label'], tf.int32)
        labels = {"hat":hat_label, "hair":hair_label, "gender":gender_label,
                  "top":top_label, "down":down_label, "shoes":shoes_label,
                  "bag":bag_label}

        return img, labels
    else:
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               "img_raw": tf.FixedLenFeature([], tf.string),
                                           })
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [128, 256, 3])
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
	#tl.visualize.frame(I=img, second=5, saveable=False, name='frame', fig_idx=12833)

        return img

if __name__ == "__main__":
    print("Prepare Data ...")
    train_indexs = load_index(is_train=True)
    test_indexs = load_index(is_train=False)
    data_to_tfrecord(train_indexs, "train_tfrecord", is_train=True)
    data_to_tfrecord(test_indexs, "test_tfrecord", is_train=False)
    print("Data Success.")
