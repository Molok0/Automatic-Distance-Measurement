import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('image/3.jpg', 0)
imgR = cv2.imread('image/4.jpg', 0)

# Создание объекта StereoBM
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Вычисление диспаритета
disparity = stereo.compute(imgL, imgR)

# Нормализация для отображения
disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# Отображение карты диспаритета
plt.imshow(disparity, 'gray')
plt.title('Disparity Map')
plt.axis('off')
plt.show()
