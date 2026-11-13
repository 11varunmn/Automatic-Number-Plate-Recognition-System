import cv2
from Recognition_Module.preprocessing import process, display_image

image = cv2.imread("./Recognition_Module/samples/image1.png")

processed_image = process(image)
display_image(image=processed_image)