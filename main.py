import mss
import numpy as np
import cv2
from time import sleep
from pydirectinput import click
import tomllib

with open("config.toml", "r") as file:
    file = tomllib.loads(file.read())

    WIDTH, HEIGHT = file['resolution'].split('x')
    WIDTH, HEIGHT = int(WIDTH), int(HEIGHT)

    ASPECT_RATIO = file['aspect_ratio']


# 1920x1080
# REGION_LMB = {'top': 434, 'left': 946, 'width': 13, 'height': 20}
# REGION_RMB = {'top': 434, 'left': 960, 'width': 13, 'height': 20}
# REGION_CIRCLE = {'top': 378, 'left': 885, 'width': 150, 'height': 150}

# 2560x1080
# REGION_LMB = {'top': 435, 'left': 1266, 'width': 13, 'height': 20}
# REGION_RMB = {'top': 435, 'left': 1281, 'width': 13, 'height': 20}
# REGION_CIRCLE = {'top': 378, 'left': 1205, 'width': 150, 'height': 150}

if ASPECT_RATIO == "16:9":
    REGION_RELATIVE_LMB = {'top': 0.40185185185185185,
                           'left': 0.49270833333333336,
                           'width': 0.0067708333333333336,
                           'height': 0.010416666666666666}
    REGION_RELATIVE_RMB = {'top': 0.40185185185185185,
                           'left': 0.50000000000000000,
                           'width': 0.0067708333333333336,
                           'height': 0.010416666666666666}
    REGION_RELATIVE_CIRCLE = {'top': 0.35,
                              'left': 0.4609375,
                              'width': 0.078125,
                              'height': 0.1388888888888889}

    # Define the center and radius of the circular region
    CENTER_X, CENTER_Y = 75, 75
    OUTER_RADIUS = 70
    INNER_RADIUS = 42
elif ASPECT_RATIO == "21:9":
    REGION_RELATIVE_LMB = {'top': 0.4027777777777778,
                           'left': 0.49453125,
                           'width': 0.005078125,
                           'height': 0.0078125}
    REGION_RELATIVE_RMB = {'top': 0.4027777777777778,
                           'left': 0.500390625,
                           'width': 0.005078125,
                           'height': 0.0078125}
    REGION_RELATIVE_CIRCLE = {'top': 0.35,
                              'left': 0.470703125,
                              'width': 0.05859375,
                              'height': 0.1388888888888889}

    # Define the center and radius of the circular region
    CENTER_X, CENTER_Y = 75, 75
    OUTER_RADIUS = 70
    INNER_RADIUS = 48
else:
    raise ValueError(f"Unsupported aspect ratio: {ASPECT_RATIO}")

REGION_LMB = {'top': REGION_RELATIVE_LMB['top'] * HEIGHT,
              'left': REGION_RELATIVE_LMB['left'] * WIDTH,
              'width': REGION_RELATIVE_LMB['width'] * WIDTH,
              'height': REGION_RELATIVE_LMB['height'] * WIDTH}
REGION_RMB = {'top': REGION_RELATIVE_RMB['top'] * HEIGHT,
              'left': REGION_RELATIVE_RMB['left'] * WIDTH,
              'width': REGION_RELATIVE_RMB['width'] * WIDTH,
              'height': REGION_RELATIVE_RMB['height'] * WIDTH}
REGION_CIRCLE = {'top': REGION_RELATIVE_CIRCLE['top'] * HEIGHT,
                 'left': REGION_RELATIVE_CIRCLE['left'] * WIDTH,
                 'width': REGION_RELATIVE_CIRCLE['width'] * WIDTH,
                 'height': REGION_RELATIVE_CIRCLE['height'] * HEIGHT}

# Convert from float to int
REGION_LMB = {k: int(v) if isinstance(v, float) else v for k, v in REGION_LMB.items()}
REGION_RMB = {k: int(v) if isinstance(v, float) else v for k, v in REGION_RMB.items()}
REGION_CIRCLE = {k: int(v) if isinstance(v, float) else v for k, v in REGION_CIRCLE.items()}

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
    elif contains_color(capture_region(REGION_RMB)):
        return 'right'


def capture_circle_region() -> np.ndarray:
    img = capture_region(REGION_CIRCLE)

    # Create a mask image with a circular region
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.circle(mask, (CENTER_X, CENTER_Y), OUTER_RADIUS, 255, -1)
    cv2.circle(mask, (CENTER_X, CENTER_Y), INNER_RADIUS, 0, -1)

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
    print(f"Starting with {WIDTH}x{HEIGHT} resolution, {ASPECT_RATIO} aspect ratio...")
    while True:
        button = get_mouse_button()

        if button and line_in_sector(capture_circle_region()):
            click(button=button)
            print(f'clicked {button}')
            sleep(0.1)

