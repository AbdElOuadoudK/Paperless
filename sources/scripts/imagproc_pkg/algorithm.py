from .utils import *
import cv2
from sklearn.cluster import HDBSCAN


class ResumeParser:
    """
    A class for processing and parsing resumes.
    
    Attributes:
        _display (bool): Determines whether to display images.
        _save (bool): Determines whether to save processed images.

    By Khouri A. Ouadoud
    """ 

    def __init__(self, save: bool = False, display: bool = False):
        """
        Initializes the ResumeParser object.

        Args:
            save (bool): Option to save processed images. Default is False.
        """
        self._save = save
        self._display = display

    def ProcessImage(self):
        pass

    def __main(self):
        pass

    def ScanFrames(self, img: numpy.ndarray, margin: int = 25, max_area: int = 30000, display: bool = True) -> tuple:
        """
        Scans the image and extracts bounding boxes by detecting edges across color channels.

        Args:
            img (numpy.ndarray): The input image to process.
            margin (int): The margin to use when merging boxes. Default is 25.
            max_area (int): Maximum allowable area for a box. Default is 30000.
            display (bool): Option to display the processed image with boxes. Default is True.

        Returns:
            tuple: The processed image and list of bounding boxes.
        """
        blue, green, red = cv2.split(img)
        blue_edges = medianCanny(blue, 0, 1)
        green_edges = medianCanny(green, 0, 1)
        red_edges = medianCanny(red, 0, 1)
        edges = blue_edges | green_edges | red_edges

        # Detect boxes in the edges image
        boxes = scan_img(edges, margin, max_area)

        # Display the image with boxes if required
        display_(img, boxes) if display else None

        return img, boxes

    def CroppeImage(self, image: numpy.ndarray, boxes: list, index: int, kernel_max_size: int = 19, padding: int = 3, display: bool = True) -> tuple:
        """
        Crops the image based on bounding boxes, processes the cropped area, and attempts to detect text lines.

        Args:
            image (numpy.ndarray): The input image from which to crop.
            boxes (list): List of bounding boxes.
            index (int): Index of the bounding box to crop.
            kernel_max_size (int): Maximum size of the kernel for line detection. Default is 19.
            padding (int): Padding for bounding boxes. Default is 3.
            display (bool): Option to display the processed image. Default is True.

        Returns:
            tuple: Cropped image and detected lines or the result of ClusterBoxes if no lines are detected.
        """
        cropped_img = retrieve_cropped(image, boxes, index)

        # Convert to grayscale, apply Gaussian blur, and binarize the image
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        all_knl_lines = []
        for k in range(1, kernel_max_size + 1):
            # Create a horizontal kernel and apply the closing operation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
            bw_closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            lines = get_lines(bw_closed, padding=padding)
            all_knl_lines += lines[:] if lines else []

        # If no lines are detected, cluster boxes
        if not any(all_knl_lines):
            print('Second FLAG')
            return self.ClusterBoxes(cropped_img)

        lines = []
        for l in all_knl_lines:
            if l and (l not in lines):
                if check_size(l[:], 52):
                    lines.append(l[:])

        # If no valid lines are detected, cluster boxes
        if not lines:
            print('Second FLAG')
            return self.ClusterBoxes(cropped_img)

        # Sort and display detected lines
        lines = iter_boxes(lines, 0)
        lines = sorted(lines, key=lambda l: (l[0][1], l[0][0]))
        print('First FLAG')
        display_(cropped_img, lines) if display else None

        return cropped_img, lines

    def ClusterBoxes(self, cropped_img: numpy.ndarray, display: bool = True) -> tuple:
        """
        Clusters bounding boxes within the cropped image using HDBSCAN clustering.

        Args:
            cropped_img (numpy.ndarray): The cropped input image.
            display (bool): Option to display the clustered boxes. Default is True.

        Returns:
            tuple: Cropped image and merged bounding boxes.
        """
        cropped_img, boxes = self.ScanFrames(cropped_img, margin=5, max_area=3000, display=False)
        unpacked_boxes = [unpack(bx) for bx in boxes]
        x_centers_ = [[x, y, h / 2] for x, y, w, h in unpacked_boxes]
        epsilon = min([item[2] for item in x_centers_])

        # Perform clustering on the bounding box centers
        clustering = HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=epsilon, n_jobs=-1)
        clustering = clustering.fit(x_centers_)
        merged_boxes = merge_clusters(unpacked_boxes, clustering.labels_)
        merged_boxes = [pack(bx) for bx in merged_boxes]

        # Display merged boxes if required
        display_(cropped_img, merged_boxes) if display else None

        return cropped_img, merged_boxes
