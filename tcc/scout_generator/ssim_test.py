from skimage.metrics import structural_similarity
import cv2
import os

print(os.getcwd())

first_frame = cv2.cvtColor(cv2.imread('../1.png'), cv2.COLOR_BGR2GRAY)
second_frame = cv2.cvtColor(cv2.imread('../2.png'), cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(first_frame, second_frame, full=True)
diff = (diff * 255).astype('uint8')

print('SSIM: {}'.format(score))