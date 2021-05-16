


import numpy as np
import matplotlib.pyplot as plt
import cv2
##
def score_pixels(img) -> np.ndarray:
        """
        도로 이미지를 촬영하고 픽셀 강도가 차선의 일부일 가능성에 매핑되는 이미지를 반환한다.
        각 픽셀은 픽셀 강도로 저장되는 자체 점수를 받는다. 0의 강도는 차선에서 오는 것이 아니라는 것을 의미한다.
        점수가 높으면 차선에서 나온 것에 대한 신뢰도가 높아진다.
        :param img: 일반적으로 오버헤드 관점에서 본 도로의 이미지.
        :반환: 점수 이미지.
        """

        # Settings to run thresholding operations on
        settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 200}, # Yellow detect, 220
                    {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 200}, #220 
                    {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 200}] #210

        # Perform binary thresholding according to each setting and combine them into one image.
        scores = np.zeros(img.shape[0:2]).astype('uint8')
        
        fig = plt.figure(3)
        ax = []
        tmp = 1

        for params in settings[1:]:
            # Change color space
            color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))
            print(type(color_t))
            print("cspace:",params['cspace'])
            imgUMat = cv2.UMat(img)
            cv2.imshow("img141516",imgUMat)
            cv2.waitKey(0)
            gray = cv2.cvtColor(imgUMat, color_t).get()
            cv2.imshow("gray",gray)
            cv2.waitKey(0)
            gray2 = gray[:, :, params['channel']]

            cv2.imshow("img",gray2)
            cv2.waitKey(0)
            print(gray2)

            # Normalize regions of the image using CLAHE, 히스토그램 균일화
            clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))

            #clahe는 clipLimit에 따라서 한쪽으로 치우친 명암을 고르게 분포시켜서 전체적으로 고르게 밝게 바꿔주는 기법

            norm_img = clahe.apply(gray2)

            res = np.hstack((gray2,norm_img))

            cv2.imshow('clahe',res)
            cv2.waitKey(0)

            #norm_img = gray


            # Threshold to binary
            ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY_INV)

            #이진화 형태의 이미지로 출력
            plt.imshow(binary, cmap="Greys")
            plt.show()

            scores += binary

        print("%%%%%%%%%%")
        plt.imshow(cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX))
        plt.show()

        return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)



img = cv2.imread('C:/data/peter_moran/road_7.jpg')
plt.figure(2)
plt.imshow(img)
plt.show()
a = score_pixels(img)
print(a)
