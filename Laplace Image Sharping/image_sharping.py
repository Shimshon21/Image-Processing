import cv2

def sharpen_image(grey_image):
    # Define parameters for Laplacian filter
    ddepth = cv2.CV_16S
    kernel_size = 3

    # Apply Gaussian blur to remove noise
    blur_image = cv2.GaussianBlur(grey_image, (3, 3), 0)

    # Apply Laplacian filter to the blurred image
    laplacian_image = cv2.Laplacian(blur_image, ddepth, ksize=kernel_size)

    # Calculate sharpened image by subtracting Laplacian from blurred image
    sharpened_image = blur_image + (-1) * laplacian_image

    # Convert the images to absolute scale (0-255)
    abs_blur = cv2.convertScaleAbs(blur_image)
    abs_laplacian = cv2.convertScaleAbs(laplacian_image)
    abs_sharpen = cv2.convertScaleAbs(sharpened_image)

    # Save the images
    cv2.imwrite("Original_image.png", grey_image)
    cv2.imwrite("Blurred_image.png", abs_blur)
    cv2.imwrite("Laplacian_Image.png", abs_laplacian)
    cv2.imwrite("Result_image.png", abs_sharpen)


if __name__ == '__main__':
    # Read the input image in grayscale
    input_image = cv2.imread("moon.JPG", 0)

    # Call the sharpen_image function with the input image
    sharpen_image(input_image)
