import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import rank

# Apply histogram equalization on a given image
def equalize_image(image, kernel_size: tuple):
    kernel = np.ones(kernel_size)
    return rank.equalize(image, footprint=kernel)


# display images befor and after enhancement
def show_images(original, processed):
    plt.subplot(221), plt.imshow(original, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(processed, cmap='gray'), plt.title('After')
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    kernel_size = (15, 15)

    image_input = cv2.imread('embedded_squares.jpg', 0)

    processed_image = equalize_image(image_input, kernel_size)

    show_images(image_input, processed_image)
