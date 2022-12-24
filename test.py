import cv2
import torch
img = cv2.imread('D:\\APPLICATIONS\\udacity-sim\\simulator-windows-64\\DATA\\IMG\\right_2022_12_16_22_29_51_314.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Is CUDA supported by this system?{torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")