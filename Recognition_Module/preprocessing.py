import cv2
import matplotlib.pyplot as plt
import math

def process(image):
    if image is None:
        print("Image not found!")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
    enhanced_image = enhance_contrast(bilateral_image)

    display_steps(
        gray_image,
        bilateral_image,
        enhanced_image
    )

    return bilateral_image


def enhance_contrast(image):

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_image = clahe.apply(image)

    return enhanced_image



def display_steps(*args):
    i=1
    for image in args:
        plt.subplot(2, math.ceil(len(args)/2), i)
        plt.imshow(image, cmap="gray")
        plt.title(f'step {i}')
        plt.axis("off") 
        i += 1
    plt.tight_layout()
    plt.show()


def display_image(image, step=None):
    if image is None:
        print("Cannot display the image!")
        return None
    cv2.imshow(step, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
