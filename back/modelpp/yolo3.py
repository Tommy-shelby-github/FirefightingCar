import colorsys
import os
import time

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo import yolo_body
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox
tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)

class YOLOMODEL(object):
    
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'logs/ep050-loss13.141-val_loss12.839.h5',
        "classes_path"      : 'model_data/cls_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        "max_boxes"         : 100,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.angle_x=0.0
        self.angle_y=1.0
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.yolo_model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))
        #---------------------------------------------------------#
        #   在DecodeBox函数中，我们会对预测结果进行后处理
        #   后处理的内容包括，解码、非极大抑制、门限筛选等
        #---------------------------------------------------------#
        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'anchors'           : self.anchors, 
                'num_classes'       : self.num_classes, 
                'input_shape'       : self.input_shape, 
                'anchor_mask'       : self.anchors_mask,
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，并进行归一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        if(len(out_boxes)):
            box             = out_boxes[0]
            top=box[0]
            left=box[1]
            bottom=box[2]
            right=box[3]

            top=top.numpy()
            left=left.numpy()
            bottom=bottom.numpy()
            right=right.numpy()
            
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            x_center = (left + right) / 2  # 中心点坐标x
            y_center = (top + bottom) / 2  # 中心点坐标y
            self.angle_x = 0.333+x_center /480  #摄像头横向看到的角度是60°，是180°的三分之一
            focalLength = 700
            knownWidth = 2
            distance = (knownWidth * focalLength) / (abs(left-right))
            list_y = [0.4,0.35,0.3,0.25,0.2,1,1,1,1,1,1,1,1,1,1]
            self.angle_y = list_y[int(distance*10/2.54)]
        else:
            self.angle_x=0.0
            self.angle_y=1.0
        return self.angle_x,self.angle_y
    

class YunTAB(object):
    '''
    allow reverse to trigger automatic reverse throttle
    '''

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.angle_x=0.0
        self.angle_y=1.0
        self.yolo = YOLOMODEL()
    def run(self,img):        
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            return self.angle_x,self.angle_y
        else:
            self.angle_x,self.angle_y = self.yolo.detect_image(image)
            return self.angle_x,self.angle_y
    #if 和else分别return，解决了问题

    
mm=YunTAB()
time_num=0
while(1):    
    
    time_num=time_num+1    
    pp,gg=mm.run('img/image_20.jpg')
    print(time_num,pp,gg)
    files=open('angle.txt','w')
    files.write(str(pp)+" "+str(gg))
    files.close
   