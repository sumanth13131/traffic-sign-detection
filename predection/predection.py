import cv2
import numpy as np
from PIL import Image
import tensorflow
MODEL=tensorflow.keras.models.load_model('cnn.model')
lis=['20','30','50','60','70','80','100','120','bicycles_crossing','bumpy_road',
     'children_crossing','dang_cur_to_left','dang_cur_to_right','double_curve','general_caution','pedestrians',
     'road_work','slippery_road','stop','traffic_signals','turn_left','turn_right','wild_animals_crossing']
###
##
#
img=cv2.imread('/test_images/bumpy_road.png',0)
cv2.imshow('frame',img)
img1=cv2.resize(img,(128,128))
print(img1.shape)
data1=np.asarray(img1,dtype="int32")
data2=data1/255.0
data3=np.array(data2).reshape(-1,128,128,1)
j=MODEL.predict([data3])
i=np.argmax(j)
print('SIGN BOARD:{0}'.format(lis[i]))
cv2.waitKey(0)
cv2.destroyAllWindows()
