# TensorFlow高效加载数据的方法
# 概述
 关于Tensorflow读取数据，官网给出了三种方法：

- **供给数据(Feeding)**： 在TensorFlow程序运行的每一步， 让Python代码来供给数据。
- **从文件读取数据**： 在TensorFlow图的起始， 让一个输入管线从文件中读取数据。
- **预加载数据**： 在TensorFlow图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况)。

对于数据量较小而言，可能一般选择直接将数据加载进内存，然后再分`batch`输入网络进行训练（tip:使用这种方法时，结合`yield` 使用更为简洁，大家自己尝试一下吧，我就不赘述了）。但是，如果数据量较大，这样的方法就不适用了，因为太耗内存，所以这时最好使用tensorflow提供的队列`queue`，也就是第二种方法 **从文件读取数据**。对于一些特定的读取，比如csv文件格式，官网有相关的描述，在这儿我介绍一种比较通用，高效的读取方法（官网介绍的少），即使用tensorflow内定标准格式——`TFRecords`

------

# TFRecords

TFRecords其实是一种二进制文件，虽然它不如其他格式好理解，但是它能更好的利用内存，更方便复制和移动，并且不需要单独的标签文件（等会儿就知道为什么了）... ...总而言之，这样的文件格式好处多多，所以让我们用起来吧。

TFRecords文件包含了`tf.train.Example` 协议内存块(protocol buffer)(协议内存块包含了字段 `Features`)。我们可以写一段代码获取你的数据， 将数据填入到`Example`协议内存块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过`tf.python_io.TFRecordWriter` 写入到TFRecords文件。

从TFRecords文件中读取数据， 可以使用`tf.TFRecordReader`的`tf.parse_single_example`解析器。这个操作可以将`Example`协议内存块(protocol buffer)解析为张量。

接下来，让我们开始读取数据之旅吧~

# 生成TFRecords文件

 我们使用`tf.train.Example`来定义我们要填入的数据格式，然后使用`tf.python_io.TFRecordWriter`来写入。

```python
import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()

'''
此处我加载的数据目录如下：
0 -- img1.jpg
     img2.jpg
     img3.jpg
     ...
1 -- img1.jpg
     img2.jpg
     ...
2 -- ...
...
'''
writer = tf.python_io.TFRecordWriter("train.tfrecords")
for index, name in enumerate(classes):
    class_path = cwd + name + "/"
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((224, 224))
        img_raw = img.tobytes()              #将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())  #序列化为字符串
writer.close()
```
 关于`Example` `Feature`的相关定义和详细内容，我推荐去官网查看相关API。

基本的，一个`Example`中包含`Features`，`Features`里包含`Feature`（这里没s）的字典。最后，`Feature`里包含有一个 `FloatList`， 或者`ByteList`，或者`Int64List`

 就这样，我们把相关的信息都存到了一个文件中，所以前面才说不用单独的label文件。而且读取也很方便。

```python
for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

	image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    print image, label
```


# 使用队列读取
 一旦生成了TFRecords文件，接下来就可以使用队列（`queue`）读取数据了。
```python
def read_and_decode(filename):
	#根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label
```

 之后我们可以在训练的时候这样使用
```python
img, label = read_and_decode("train.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(3):
        val, l= sess.run([img_batch, label_batch])
        #我们也可以根据需要对val， l进行处理
        #l = to_categorical(l, 12)
        print(val.shape, l)
```
 至此，tensorflow高效从文件读取数据差不多完结了。


 恩？等等...什么叫差不多？对了，还有几个**注意事项**：

 第一，tensorflow里的graph能够记住状态（`state`），这使得`TFRecordReader`能够记住`tfrecord`的位置，并且始终能返回下一个。而这就要求我们在使用之前，必须初始化整个graph，这里我们使用了函数`tf.initialize_all_variables()`来进行初始化。

第二，tensorflow中的队列和普通的队列差不多，不过它里面的`operation`和`tensor`都是符号型的（`symbolic`），在调用`sess.run()`时才执行。

 第三， `TFRecordReader`会一直弹出队列中文件的名字，直到队列为空。

------
# 总结

 1. 生成tfrecord文件
 2. 定义`record reader`解析tfrecord文件
 3. 构造一个批生成器（`batcher`）
 4. 构建其他的操作
 5. 初始化所有的操作
 6. 启动`QueueRunner`
 7. 运行训练循环
