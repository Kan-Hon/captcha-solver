import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import json

class Captcha(object):
    def __init__(self, hu_moments_json_path):
        self.hu_moment_dict_mean = json.load(open(hu_moments_json_path))

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        img  = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img)
        ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        
        roi_list = self.get_roi_from_image(thresh)
        results = []
        # Calculate Moments 
        for i in range(len(roi_list)):
            roi = roi_list[i]
            hu_moments = self.get_moments_from_roi(roi, thresh)
            result = self.get_char_nearest_hu(hu_moments)
            results.append(result)
            
        self.save_to_txt_file(save_path, ''.join(results))
        
        
    def get_roi_from_image(self, thresh):

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        roi_list = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            roi_list.append((x, y, w, h))

        roi_list.sort(key=lambda x:x[0])
        return roi_list
    
    def get_moments_from_roi(self, roi, thresh):
        (x, y, w, h) = roi
        left, top = x-1, y-1
        right, bottom = x+w+1, y+h+1
        roi_image = thresh[top:top+h+2, left:left+w+2]
        # plt.imshow(roi_image)
        # plt.show()
        moments = cv2.moments(roi_image) 
        # Calculate Hu Moments 
        huMoments = cv2.HuMoments(moments)
        huMoments = huMoments.flatten()
        # for i in range(len(huMoments)):
            # print(huMoments[i])
            # huMoments[i] = (huMoments[i]/huMoments[i]+0.0001) *  math.log(abs(huMoments[i]),10)
        return huMoments

    def get_char_nearest_hu(self, hu_moments):
        nearest_dist = np.Inf
        char = None
        for k,v in self.hu_moment_dict_mean.items():
            dist = np.linalg.norm(hu_moments - self.hu_moment_dict_mean[k])
            if dist < nearest_dist:
                char = k
                nearest_dist = dist
        return char
    def save_to_txt_file(self, save_path, result):
        with open(save_path, 'w') as f:
            f.write(result)

if __name__ == "__main__":
    cap = Captcha('./hu_moments_values.json')
    cap('./sampleCaptchas/input/input01.jpg', './result.txt')