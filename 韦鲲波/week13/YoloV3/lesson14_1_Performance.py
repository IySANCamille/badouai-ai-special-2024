import os
import random
import colorsys
import numpy as np
import lesson14_1_config as config
import lesson14_1_utils as utils
import lesson14_1_Model as Model
import tensorflow.compat.v1 as tf
from PIL import Image, ImageFont, ImageDraw

tf.disable_eager_execution()
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def predict(image):
    # 图片处理
    DP_img = utils.DataPreprocessor(image)
    src_img = DP_img.Img
    print(src_img.size)
    prep_img = DP_img.image_preprocess()

    # 创建变量
    inp_img = tf.placeholder(tf.float32, shape=[None, 416, 416, 3])
    inp_img_shape = tf.placeholder(tf.float32, shape=[2,])

    # 初始化后处理对象
    postprocess = utils.DataPostprocessor(
        config.obj_threshold,
        config.nms_threshold,
        config.classes_path,
        config.anchors_path
    )

    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            with tf.variable_scope('predict'):
                model = Model.yolo(
                    config.norm_epsilon,
                    config.norm_decay,
                    config.anchors_path,
                    config.classes_path,
                    pre_train = False
                )

                output = model.yolo_inference(
                    inp_img,
                    config.num_anchors // 3,
                    config.num_classes,
                    training=False
                )

                boxes, scores, classes = postprocess.yolo3_postprocessor(
                    output,
                    inp_img_shape,
                    max_bboxes=20
                )

                weights_load = utils.Weights_load(
                    var_list=tf.global_variables(scope = 'predict'),
                    yolo3_weights=config.yolo3_weights_path
                ).weights_load()

                sess.run(weights_load)

                # 进行预测
                out_boxes, out_scores, out_classes = sess.run(
                    [boxes, scores, classes],
                    feed_dict={
                        # image_data这个resize过
                        inp_img: prep_img,
                        # 以y、x的方式传入
                        inp_img_shape: [src_img.size[1], src_img.size[0]]
                    })




                # ---------------------------------------#
                #   画框
                # ---------------------------------------#
                # 找到几个box，打印
                print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
                font = ImageFont.truetype(font='yolo3-tensorflow-master/font/FiraMono-Medium.otf',
                                          size=np.floor(3e-2 * src_img.size[1] + 0.5).astype('int32'))

                # 厚度
                thickness = (src_img.size[0] + src_img.size[1]) // 300

                for i, c in reversed(list(enumerate(out_classes))):
                    # 获得预测名字，box和分数
                    predicted_class = postprocess.class_names[c]
                    box = out_boxes[i]
                    score = out_scores[i]

                    # 打印
                    label = '{} {:.2f}'.format(predicted_class, score)

                    # 用于画框框和文字
                    draw = ImageDraw.Draw(src_img)
                    # 获得写字的时候，按照这个字体，要多大的框
                    label_size = draw.textbbox((0, 0), label, font=font)
                    label_width = label_size[2] - label_size[0]
                    label_height = label_size[3] - label_size[1]

                    # 获得四个边
                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(src_img.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
                    right = min(src_img.size[0] - 1, np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))
                    print(label_size)

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_height])
                    else:
                        text_origin = np.array([left, top + 1])

                    # My kingdom for a good redistributable image drawing library.
                    for i in range(thickness):
                        draw.rectangle(
                            [left + i, top + i, right - i, bottom - i],
                            outline=postprocess.colors[c])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + [label_width, label_height])],
                        fill=postprocess.colors[c])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw
                src_img.show()
                src_img.save('yolo3-tensorflow-master/img/result1.jpg')



if __name__ == '__main__':
    predict(config.image_file)

















