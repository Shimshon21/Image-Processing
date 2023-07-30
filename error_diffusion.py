import cv2
import numpy as np


def color_values(m):
    """
    Generate a list of equally spaced color values ranging from 0 to 255.

    Args:
        m (int): The number of colors required.

    Returns:
        list: A list of color values ranging from 0 to 255.
    """
    temp_color_values = []
    for i in range(m):
        # Calculate the color value for the current step and limit it to the range [0, 255].
        temp_color_values.append(np.clip(i * round(255 / (m - 1)), 0, 255))
    return temp_color_values


def get_new_pixel_value(pixel, colors_values):
    """
    Map the input pixel value to the corresponding color value based on the thresholds.

    Args:
        pixel (int): The pixel value to map to a color.
        colors_values (list): A list of color values.

    Returns:
        int: The color value corresponding to the input pixel value.
    """
    new_pixel = 0

    for i in range(len(colors_values) - 1):
        color_start = colors_values[i]
        color_end = colors_values[i + 1]

        color_range_threshold = round((color_start + color_end) / 2)

        if pixel >= color_range_threshold:
            new_pixel = color_end
        else:
            break

    return new_pixel


def error_diffusion_by_levels_with_dynamic_threshold(grayscale, m):
    """
     Apply error diffusion with dynamic thresholding to a grayscale image.

     Args:
         grayscale (numpy.ndarray): A 2D array representing the grayscale image.
         m (int): The number of levels (colors) required for the output image.

     Returns:
         image_output: The processed grayscale image after error diffusion.
     """
    height, width = grayscale.shape
    image_output = np.copy(grayscale)
    error_array = np.zeros((height, width))

    # Calculate the threshold as half of the distance between each adjacent level
    colors_values = color_values(m)

    for y in range(height - 1):
        for x in range(1, width - 1):
            old_pixel = grayscale[y, x]

            new_pixel = get_new_pixel_value(old_pixel + error_array[y, x], colors_values)

            image_output[y, x] = new_pixel

            # Calculate the error
            diff = (grayscale[y, x] + error_array[y, x]) - new_pixel

            # Diffuse the error to neighboring pixels
            error_array[y + 1, x] = error_array[y + 1, x] + diff * 3 / 8
            error_array[y + 1, x + 1] = error_array[y + 1, x + 1] + diff * 1 / 4
            error_array[y, x + 1] = error_array[y, x + 1] + diff * 3 / 8

    return image_output


if __name__ == '__main__':
    grey_image = cv2.imread("lena.png", 0)

    enhanced_error_diffusion_image = error_diffusion_by_levels_with_dynamic_threshold(grey_image, 2)
    enhanced_error_diffusion_image2 = error_diffusion_by_levels_with_dynamic_threshold(grey_image, 3)
    enhanced_error_diffusion_image3 = error_diffusion_by_levels_with_dynamic_threshold(grey_image, 4)
    enhanced_error_diffusion_image4 = error_diffusion_by_levels_with_dynamic_threshold(grey_image, 5)

    cv2.imwrite("images/error diffusion 2.png", enhanced_error_diffusion_image)
    cv2.imwrite("images/error diffusion 3.png", enhanced_error_diffusion_image2)
    cv2.imwrite("images/error diffusion 4.png", enhanced_error_diffusion_image3)
    cv2.imwrite("images/error diffusion 5.png", enhanced_error_diffusion_image4)
