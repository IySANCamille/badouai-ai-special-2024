import os
import random
import colorsys
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image, ImageFont, ImageDraw


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.Img = Image.open(self.file_path)

    # 图片预处理，缩放，整合为416
    def image_preprocess(self, tar_size=416):
        # 获取图片的尺寸
        img_w, img_h = self.Img.size
        # 计算长边缩放到416的比例
        scale = min(tar_size / img_h, tar_size / img_w)

        # 求得缩放后的尺寸，此时需要int一下，因为resize的size需要接受整形的参数
        cur_h, cur_w = int(img_h * scale), int(img_w * scale)
        # 将图片按缩放的尺寸缩放
        tar_img = self.Img.resize((cur_w, cur_h), resample=Image.BICUBIC)

        # 创建一个用于粘贴的底板
        integration_img = Image.new('RGB', (tar_size, tar_size), color=(128, 128, 128))
        # 将刚刚缩放后的图放在底板中心，创造出416*416的图形
        integration_img.paste(tar_img, ((tar_size - cur_w) // 2, (tar_size - cur_h) // 2))

        # 因为image对象没有shape方法，但是有size方法可以看形状
        print(type(integration_img), integration_img.size)

        # 格式改为np数组
        integration_img = np.array(integration_img)
        # 归一化，因为是要有小数，所以得改为float32格式，在np.array时就可以设置dtype类型，只是为了易读在这里单独写了
        integration_img = integration_img.astype('float32')
        integration_img /= 255.
        # 增加一个batch轴
        integration_img = np.expand_dims(integration_img, axis=0)

        print(type(integration_img), integration_img.shape)

        # 返回的结果是numpy类型4维的数组（张量）
        return integration_img


class Weights_load:
    def __init__(self, var_list, yolo3_weights):
        self.var_list = var_list
        self.yolo3_weights = yolo3_weights

    # 权重读取
    def weights_load(self):
        # 读取权重文件
        with open(self.yolo3_weights, "rb") as fp:
            # 摒弃一些头文件信息
            _ = np.fromfile(fp, dtype=np.int32, count=5)
            # 读取头文件信息之后的权重数值
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        i = 0
        assign_ops = []
        while i < len(self.var_list) - 1:

            var1 = self.var_list[i]
            var2 = self.var_list[i + 1]

            # 读取了很多当前模型的权重，其中为很多层定义了名称，这里筛选名称包含conv2d的层
            if 'conv2d' in var1.name.split('/')[-2]:
                # check type of next layer
                if 'batch_normalization' in var2.name.split('/')[-2]:
                    # load batch norm params
                    gamma, beta, mean, var = self.var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                    # we move the pointer by 4, because we loaded 4 variables
                    i += 4
                elif 'conv2d' in var2.name.split('/')[-2]:
                    # load biases
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                    ptr += bias_params
                    assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                    # we loaded 1 variable
                    i += 1
                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1

        return assign_ops


class DataPostprocessor:
    def __init__(self, obj_threshold, nms_threshold, classes_file, anchors_file):
        # 对yolo3输出的结果进行处理时需要用到的阈值
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold

        # 用于置信度评分的分类和先验框答案
        self.classes_path = classes_file
        self.anchors_path = anchors_file

        # 初始化对象时先把分类和先验框读取
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        # 画框框用
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]

        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)


    #-------------------------------------#
    #   读取分类的方法
    def _get_class(self):
        """
        Introduction
        ------------
            读取类别名称
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    #-------------------------------------#
    #   读取先验框的方法
    def _get_anchors(self):
        """
        Introduction
        ------------
            读取anchors数据
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors


    #-------------------------------------#
    #   获取原图框坐标的方法
    def _correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # 416,416
        input_shape = tf.cast(input_shape, dtype = tf.float32)
        # 实际图片的大小
        image_shape = tf.cast(image_shape, dtype = tf.float32)

        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))

        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        return boxes


    # -------------------------------------#
    #   boxes_and_scores中获取框信息的方法
    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            num_classes: 类别数量
            input_shape: 输入大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])

        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)

        # 将x,y坐标归一化，相对网格的位置
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs


    # -------------------------------------#
    #   获取框坐标和置信度的方法
    def _boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        # 获得特征
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        # 寻找在原图上的位置
        boxes = self._correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        # 获得置信度box_confidence * box_class_probs
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores

    # -------------------------------------#
    #   对结果进行后处理，获得最终结果的框坐标、分数、分类
    def yolo3_postprocessor(self, result, src_img_shape, max_bboxes=20):
        # 初始化先验框
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # 初始化解码后的bbox坐标列表
        bboxes = []
        # 初始化每个bbox的分数列表
        bbox_scores = []

        input_shape = tf.shape(result[0])[1: 3] * 32

        # 读取并分类yolo3输出的结果
        for i, _ in enumerate(result):
            bbox, scores = self._boxes_and_scores(
                result[i],
                self.anchors[anchor_mask[i]],
                len(self.class_names),
                input_shape,
                src_img_shape
            )
            bboxes.append(bbox)
            bbox_scores.append(scores)

        # 将各结果concat到一起
        bboxes = tf.concat(bboxes, axis=0)
        bbox_scores = tf.concat(bbox_scores, axis=0)

        # 创建一个掩码，用于检测哪些bbox的分大于阈值
        mask = bbox_scores >= self.obj_threshold
        max_bboxes = tf.constant(max_bboxes, dtype=tf.int32)

        # 初始化三个列表，用于存储最终的边界框、得分、分类
        boxes_num = []
        scores_num = []
        classes_num = []

        # 提取出每个分类进行nms后的结果，汇总
        for i, _ in enumerate(self.class_names):
            # 在concat完的那个bboxes中，根据创建的mask规则进行筛选，取出所有类为i的box
            class_boxes = tf.boolean_mask(bboxes, mask[:, i])
            # 取出所有类为i的分数
            class_box_scores = tf.boolean_mask(bbox_scores[:, i], mask[:, i])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(
                class_boxes,
                class_box_scores,
                max_bboxes,
                iou_threshold=self.nms_threshold
            )

            # 获取非极大抑制的结果
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * i

            boxes_num.append(class_boxes)
            scores_num.append(class_box_scores)
            classes_num.append(classes)

        boxes_num = tf.concat(boxes_num, axis=0)
        scores_num = tf.concat(scores_num, axis=0)
        classes_num = tf.concat(classes_num, axis=0)
        return boxes_num, scores_num, classes_num


if __name__ == '__main__':
    a = DataPreprocessor('yolo3-tensorflow-master/img/img.jpg')
    a.image_preprocess()


