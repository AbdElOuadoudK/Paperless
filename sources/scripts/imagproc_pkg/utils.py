import scipy, numpy, pandas
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pytesseract


def display(img: numpy.ndarray = None, img_path: str = None) -> None:
    """
    Display an image either from a numpy array or from a file path.

    Args:
        img (numpy.ndarray): The image to display as a numpy array. Default is None.
        img_path (str): The path to the image file. Default is None.

    Raises:
        ValueError: If neither `img` nor `img_path` is provided.

    Returns:
        None: Displays the image using matplotlib.
    """
    if img is None:
        if img_path is not None:
            # Read the image from the provided file path using OpenCV.
            img = cv2.imread(img_path)
        else:
            raise ValueError("No image provided. Please provide either 'img_path' or 'img'.")

    if len(img.shape) == 2:  # Image is in grayscale
        height, width = img.shape
        depth = 1  # Grayscale images have a single channel
    else:  # Image is in color
        height, width, depth = img.shape
        # Convert image from BGR (used by OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    figsize = width / float(80), height / float(80)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')  # Hide axis
    ax.imshow(img, cmap='gray' if depth == 1 else None)
    plt.show()


def writeDisplay(img: numpy.ndarray, img_path: str) -> None:
    """
    Save an image to a specified path and display it.

    Args:
        img (numpy.ndarray): The image to save.
        img_path (str): The path where the image will be saved.

    Returns:
        None: Saves and displays the image.
    """
    cv2.imwrite(img_path, img)
    display(img_path=img_path)


def greyscale(img: numpy.ndarray) -> numpy.ndarray:
    """
    Convert a color image to grayscale.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The grayscale image.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def thinning(img: numpy.ndarray) -> numpy.ndarray:
    """
    Perform image thinning using erosion.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The thinned image.
    """
    kernel = numpy.ones((2, 2), numpy.uint8)
    img_copy = cv2.erode(img, kernel, iterations=1)
    return img_copy


def noise_removal(img: numpy.ndarray) -> numpy.ndarray:
    """
    Remove noise from an image using morphological transformations.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The image with reduced noise.
    """
    kernel = numpy.ones((1, 1), numpy.uint8)
    img_copy = cv2.dilate(img, kernel, iterations=3)
    # Perform closing operation and apply median blur
    img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)
    img_copy = cv2.medianBlur(img_copy, 3)
    return img_copy


def tup(point: tuple) -> tuple:
    """
    Convert a point to a tuple.

    Args:
        point (tuple): A point as a list or other iterable.

    Returns:
        tuple: The point as a tuple.
    """
    return (point[0], point[1])


def medianCanny(img: numpy.ndarray, thresh1: float, thresh2: float) -> numpy.ndarray:
    """
    Apply the Canny edge detection algorithm using median values.

    Args:
        img (numpy.ndarray): The input image.
        thresh1 (float): The lower threshold factor.
        thresh2 (float): The upper threshold factor.

    Returns:
        numpy.ndarray: The image with Canny edge detection applied.
    """
    img_copy = img.copy()
    median = numpy.median(img_copy)
    img_copy = cv2.Canny(img_copy, int(thresh1 * median), int(thresh2 * median))
    return img_copy


def display_(img: numpy.ndarray, boxes: list) -> numpy.ndarray:
    """
    Display an image with bounding boxes.

    Args:
        img (numpy.ndarray): The input image.
        boxes (list): List of bounding box coordinates.

    Returns:
        numpy.ndarray: The image with bounding boxes drawn.
    """
    img_copy = img.copy()
    for box in boxes:
        cv2.rectangle(img_copy, tup(box[0]), tup(box[1]), (0, 255, 0), 1)
    display(img=img_copy)

    return img_copy

def overlap_1(source: list, target: list) -> bool:
    """
    Check if two bounding boxes overlap.

    Args:
        source (list): The first bounding box, specified as [[top-left], [bottom-right]].
        target (list): The second bounding box, specified as [[top-left], [bottom-right]].

    Returns:
        bool: True if the two boxes overlap, False otherwise.
    """
    tl1, br1 = source
    tl2, br2 = target

    # Check for non-overlapping conditions
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True


def overlap_2(source: list, target: list) -> bool:
    """
    Check if two bounding boxes overlap with an additional height check.

    Args:
        source (list): The first bounding box, specified as [[top-left], [bottom-right]].
        target (list): The second bounding box, specified as [[top-left], [bottom-right]].

    Returns:
        bool: True if the two boxes overlap, False otherwise.
    """
    tl1, br1 = source
    tl2, br2 = target
    avg_h = (br1[1] - tl1[1]) / 3

    # Check for non-overlapping conditions, including height difference
    if abs(br1[1] - br2[1]) > avg_h:
        return False
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False
    return True


def getAllOverlaps(boxes: list, box: list, index: int, overlap_func=overlap_1) -> list:
    """
    Get all overlapping boxes for a given box.

    Args:
        boxes (list): List of all boxes to check for overlaps.
        box (list): The box to check overlaps against.
        index (int): The index of the current box in the list.
        overlap_func (callable): Function to check overlaps. Default is overlap_1.

    Returns:
        list: List of indices of overlapping boxes.
    """
    overlaps = []
    for idx in range(len(boxes)):
        if idx != index:
            if overlap_func(box, boxes[idx]):
                overlaps.append(idx)
    return overlaps


def iter_boxes(boxes: list, margin: int, overlap_func=overlap_1) -> list:
    """
    Iterate through bounding boxes and merge overlapping ones.

    Args:
        boxes (list): List of bounding boxes.
        margin (int): Margin for expanding the bounding box during merging.
        overlap_func (callable): Function to check overlaps. Default is overlap_1.

    Returns:
        list: List of merged bounding boxes.
    """
    boxes_copy = boxes.copy()
    finished = False
    while not finished:
        finished = True
        index = len(boxes_copy) - 1
        while index >= 0:
            curr = boxes_copy[index]
            tl = curr[0][:]
            br = curr[1][:]
            tl[0] -= margin
            tl[1] -= margin
            br[0] += margin
            br[1] += margin

            overlaps = getAllOverlaps(boxes_copy, [tl, br], index, overlap_func=overlap_func)

            if len(overlaps) > 0:
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    tl, br = boxes_copy[ind]
                    con.append([tl])
                    con.append([br])
                con = numpy.array(con)

                x, y, w, h = cv2.boundingRect(con)
                w -= 1
                h -= 1
                merged = [[x, y], [x + w, y + h]]

                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes_copy[ind]
                boxes_copy.append(merged)

                finished = False
                break
            index -= 1
    return boxes_copy


def attach_lines_(lines: list) -> list:
    """
    Attach overlapping or closely aligned lines into a single line.

    Args:
        lines (list): List of lines, each specified as [[top-left], [bottom-right]].

    Returns:
        list: List of merged lines.
    """
    lines_ = []
    lines_.append(lines[0])
    new_ = False

    for i in range(len(lines) - 1):
        if new_:
            curr_line = new_line
        else:
            curr_line = lines[i]
            new_ = True

        next_line = lines[i + 1]
        avg_h = ((curr_line[1][1] - curr_line[0][1]) + (next_line[1][1] - next_line[0][1])) / 2

        # Adjust the coordinates based on line alignment and merge conditions
        if (curr_line[0][0] >= next_line[1][0]) and (abs(curr_line[0][1] - next_line[0][1]) <= avg_h):
            curr_line = lines[i + 1]
            next_line = lines[i]

        if abs(curr_line[0][1] - next_line[0][1]) <= avg_h / 2:
            x1 = curr_line[0][0]
            y1 = min(curr_line[0][1], next_line[0][1])
            x2 = next_line[1][0]
            y2 = max(curr_line[1][1], next_line[1][1])
            new_line = [[x1, y1], [x2, y2]]
            if new_:
                lines_.pop()
            lines_.append(new_line[:])
            new_ = True
        else:
            if not new_:
                lines_.append(curr_line)
            else:
                lines_.append(next_line)
            new_ = False
    return lines_


def get_lines(img: numpy.ndarray, padding: int = 3) -> list:
    """
    Extract text lines from an image using contours.

    Args:
        img (numpy.ndarray): The input image.
        padding (int): Padding to apply to the bounding boxes. Default is 3.

    Returns:
        list: List of bounding boxes representing lines.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if (cv2.boundingRect(cnt)[2] / cv2.boundingRect(cnt)[3]) >= 2.0]

    if len(filtered_contours) == 0:
        return None

    sorted_contours = sorted(filtered_contours, key=lambda contour: (cv2.boundingRect(contour)[1], cv2.boundingRect(contour)[0]))

    lines = []  # List to store line bounding boxes
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = (x - padding, y - padding, w + padding, h + padding)

        if 2 * padding < h:
            lines.append([[x, y], [x + w, y + h]])

    return lines


unpack = lambda x: (x[0][0], x[0][1], x[1][0] - x[0][0], x[1][1] - x[0][1])
pack = lambda x: [[x[0], x[1]], [x[0] + x[2], x[1] + x[3]]]


def check_size(l: list, size: int) -> bool:
    """
    Check if a box exceeds a given size.

    Args:
        l (list): Bounding box as [[top-left], [bottom-right]].
        size (int): Minimum size threshold.

    Returns:
        bool: True if the box is larger than the size, False otherwise.
    """
    ul = unpack(l[:])
    return ul[2] * ul[3] > size


def merge_clusters(boxes: list, labels: list) -> list:
    """
    Merge bounding boxes based on their cluster labels.

    Args:
        boxes (list): List of bounding boxes.
        labels (list): List of cluster labels corresponding to each bounding box.

    Returns:
        list: List of merged bounding boxes.
    """
    clusters = {}
    for label, box in zip(labels, boxes):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(box)

    merged_boxes = []
    for cluster in clusters.values():
        if len(cluster) > 0:
            x_min = min([box[0] for box in cluster])
            y_min = min([box[1] for box in cluster])
            x_max = max([box[0] + box[2] for box in cluster])
            y_max = max([box[1] + box[3] for box in cluster])
            merged_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return merged_boxes


def get_boxes(img: numpy.ndarray) -> list:
    """
    Extract bounding boxes from an image using contours.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        list: List of bounding boxes.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    boxes = [pack(box) for box in boxes]
    return boxes

   
def scan_img(img: numpy.ndarray, merge_margin: int, max_area: int) -> list:
    """
    Scan the image and return the bounding boxes after filtering and merging.

    Args:
        img (numpy.ndarray): The input image to scan for bounding boxes.
        merge_margin (int): The margin to use when merging overlapping boxes.
        max_area (int): The maximum area of a box to consider, larger boxes are filtered out.

    Returns:
        list: List of filtered and merged bounding boxes.
    """
    boxes = get_boxes(img)
    
    # Filter out boxes that exceed the maximum area
    filtered = []
    for box in boxes:
        w = box[1][0] - box[0][0]
        h = box[1][1] - box[0][1]
        
        if w * h < max_area:
            filtered.append(box)
    
    # Merge overlapping boxes
    boxes = iter_boxes(filtered, merge_margin)
    
    return boxes


def retrieve_cropped(img: numpy.ndarray, boxes: list, idx: int) -> numpy.ndarray:
    """
    Retrieve the cropped portion of the image for a given bounding box.

    Args:
        img (numpy.ndarray): The input image from which to crop.
        boxes (list): List of bounding boxes as [[top-left], [bottom-right]] coordinates.
        idx (int): The index of the bounding box to crop.

    Returns:
        numpy.ndarray: The cropped image corresponding to the specified bounding box.
    """
    temp = boxes[idx]
    cropped_img = img[temp[0][1]:temp[1][1], temp[0][0]:temp[1][0]]
    
    return cropped_img
