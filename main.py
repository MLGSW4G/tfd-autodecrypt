import mss
import mss.tools
import numpy as np
import cv2
from time import sleep
from pydirectinput import click

REGION_LMB = {'top': 434, 'left': 946, 'width': 13, 'height': 20}
REGION_RMB = {'top': 434, 'left': 960, 'width': 13, 'height': 20}
REGION_CIRCLE = {'top': 378, 'left': 885, 'width': 150, 'height': 150}
COLOR_LOWER = np.array([80, 200, 200])
COLOR_UPPER = np.array([100, 255, 255])


def capture_region(box: dict) -> np.ndarray:
    sct = mss.mss()
    sct_img = sct.grab(box)
    return np.array(sct_img)


def contains_color(img: np.ndarray) -> bool:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
    if cv2.countNonZero(mask) > 0:
        return True
    return False


def get_mouse_button() -> str | None:
    if contains_color(capture_region(REGION_LMB)):
        return 'left'
    else:
        if contains_color(capture_region(REGION_RMB)):
            return 'right'


def capture_circle_region() -> np.ndarray:
    img = capture_region(REGION_CIRCLE)

    # Define the center and radius of the circular region
    center_x, center_y = 75, 75
    outer_radius = 70
    inner_radius = 42

    # Create a mask image with a circular region
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), outer_radius, 255, -1)
    cv2.circle(mask, (center_x, center_y), inner_radius, 0, -1)

    # Apply the mask to the original image
    cropped_img = cv2.bitwise_and(img, img, mask=mask)

    # Crop the image to the circular region
    x, y, w, h = cv2.boundingRect(mask)
    cropped_img = cropped_img[y:y + h, x:x + w]

    return cropped_img


def line_in_sector(img: np.array) -> bool:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)

    # Find the contours of the cyan sector
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If the cyan sector is split by the red line, then the red line is inside the cyan sector
    if len(contours) > 1:
        return True
    return False


if __name__ == "__main__":
    while True:
        button = get_mouse_button()

        if button and line_in_sector(capture_circle_region()):
            click(button=button)
            print(f'clicked {button}')
            sleep(0.1)
