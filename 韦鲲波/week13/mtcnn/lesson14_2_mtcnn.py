from keras.layers import Conv2D, Input,MaxPool2D, Reshape,Activation,Flatten, Dense, Permute
from keras.layers import PReLU
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
import lesson14_2_utils as utils
import cv2

class MTCNN:
    def __init__(self):
        self.Pnet = self._Pnet('mtcnn-keras-master/model_data/pnet.h5')
        self.Rnet = self._Rnet('mtcnn-keras-master/model_data/rnet.h5')
        self.Onet = self._Onet('mtcnn-keras-master/model_data/onet.h5')


    def _Pnet(self, weights):
        input = Input(shape=(None, None, 3))

        x = Conv2D(kernel_size=3, filters=10, name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
        x = MaxPool2D(pool_size=2)(x)

        x = Conv2D(kernel_size=3, filters=16, name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

        x = Conv2D(kernel_size=3, filters=32, name='conv3')(x)
        x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

        cls = Conv2D(kernel_size=1, filters=2, activation='softmax', name='conv4-1')(x)
        bbox_reg = Conv2D(kernel_size=1, filters=4, name='conv4-2')(x)
        # fll = Conv2D(kernel_size=1, filters=10)(x)

        model = Model(inputs=input, outputs=[cls, bbox_reg])
        model.load_weights(weights, by_name=True)
        return model


    def _Rnet(self, weights):
        input = Input(shape=(24, 24, 3))

        x = Conv2D(kernel_size=3, filters=28, name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = Conv2D(kernel_size=3, filters=48, name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(kernel_size=2, filters=64, name='conv3')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

        # 老师的习惯，源代码有个换维度的操作
        x = Permute((3, 2, 1))(x)
        x = Flatten()(x)

        x = Dense(128, name='conv4')(x)
        x = PReLU(name='prelu4')(x)

        cls = Dense(2, activation='softmax', name='conv5-1')(x)
        bbox_reg = Dense(4, name='conv5-2')(x)
        # fll = Dense(10)(x)

        model = Model(inputs=input, outputs=[cls, bbox_reg])
        model.load_weights(weights, by_name=True)
        return model


    def _Onet(self, weights):
        input = Input(shape=(48, 48, 3))

        x = Conv2D(kernel_size=3, filters=32, name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = Conv2D(kernel_size=3, filters=64, name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(kernel_size=3, filters=64, name='conv3')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
        x = MaxPool2D(pool_size=2)(x)

        x = Conv2D(kernel_size=2, filters=128, name='conv4')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

        # 老师的习惯，源代码有个换维度的操作
        x = Permute((3, 2, 1))(x)
        x = Flatten()(x)

        x = Dense(256, name='conv5')(x)
        x = PReLU(name='prelu5')(x)

        cls = Dense(2, activation='softmax', name='conv6-1')(x)
        bbox_reg = Dense(4, name='conv6-2')(x)
        fll = Dense(10, name='conv6-3')(x)

        model = Model(inputs=input, outputs=[cls, bbox_reg, fll])
        model.load_weights(weights, by_name=True)
        return model


    def process(self, img, threshold):
        # -1到1均值话
        nor_img = (img.copy() - 127.5) / 127.5
        # 获取图像宽高
        height, width, _ = nor_img.shape

        # 送入工具函数，计算得出比例组
        scales = utils.calculateScales(img)

        # 定义一个存储Pnet结果的列表
        P_box = []
        # 循环把图片按比例缩放，送入P模型得出结果，存入列表
        for scale in scales:
            h = int(height * scale)
            w = int(width * scale)
            scale_img = cv2.resize(nor_img, (w, h))
            input = scale_img.reshape(1, *scale_img.shape)
            P_box.append(self.Pnet.predict(input))

        # 定义一个存储方框的列表
        P_result = []

        for i, c in enumerate(scales):
            # 有人脸的概率
            cls = P_box[i][0][0][:, :, 1]
            # 对应的框位置
            roi = P_box[i][1][0]

            # 取出每个缩放后的长宽
            h_, w_ = cls.shape
            # 取长边
            out_side = max(h_, w_)

            # 解码过程
            box = utils.detect_face_12net(
                cls,
                roi,
                out_side,
                1 / c,
                width,
                height,
                threshold[0]  # 此时的阈值是在NMS之前，用于直接去除过小的值
            )
            P_result.extend(box)

        # NMS，此时的阈值是非极大值
        P_result = utils.NMS(P_result, 0.7)

        if len(P_result) == 0:
            print('P_result is empty')
            return P_result


        #--------------------------------------------#
        # RNet
        #--------------------------------------------#

        # 定义一个Rnet的结果列表
        R_box = []

        for i in P_result:
            crop_img = nor_img[
                int(i[1]):int(i[3]),
                int(i[0]):int(i[2])
            ]
            scale_img = cv2.resize(crop_img, (24, 24))
            R_box.append(scale_img)

        R_box = np.array(R_box)
        R_predict = self.Rnet.predict(R_box)

        cls = np.array(R_predict[0])
        roi = np.array(R_predict[1])

        R_result = utils.filter_face_24net(
            cls,
            roi,
            P_result,
            width,
            height,
            threshold[1]
        )

        if len(R_result) == 0:
            print('R_result is empty')
            return R_result

        #--------------------------------------------#
        # ONet
        #--------------------------------------------#

        # 定义一个Onet的结果列表
        O_box = []

        for i in R_result:
            crop_img = nor_img[
                int(i[1]):int(i[3]),
                int(i[0]):int(i[2])
            ]
            scale_img = cv2.resize(crop_img, (48, 48))
            O_box.append(scale_img)

        O_box = np.array(O_box)
        O_predict = self.Onet.predict(O_box)

        cls = O_predict[0]
        roi = O_predict[1]
        pts = O_predict[2]

        O_result = utils.filter_face_48net(
            cls,
            roi,
            pts,
            R_result,
            width,
            height,
            threshold[2]
        )

        return O_result


if __name__ == '__main__':
    threshold = [0.5, 0.6, 0.7]
    img = cv2.imread('mtcnn-keras-master/img/test1.jpg')
    mtcnn = MTCNN()
    result = mtcnn.process(img, threshold)
    draw = img.copy()
    print(result)
    for i in result:
        if i is not None:
            W = -int(i[0]) + int(i[2])
            H = -int(i[1]) + int(i[3])
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            crop_img = img[int(i[1] + paddingH):int(i[3] - paddingH),
                       int(i[0] - paddingW):int(i[2] + paddingW)]
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            cv2.rectangle(draw, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])),
                          (255, 0, 0), 1)

            for j in range(5, 15, 2):
                cv2.circle(draw, (int(i[j + 0]), int(i[j + 1])), 2, (0, 255, 0))

    cv2.imwrite("mtcnn-keras-master/img/out.jpg", draw)

    cv2.imshow("test", draw)
    c = cv2.waitKey(0)


















