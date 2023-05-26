import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the pixel values in the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    files = glob.glob(dir_name + '/*/*/*.jpg')

    sum = np.empty((0,))
    for image_path in files: 
        img = Image.open(image_path).convert('L')
        arr = np.array(img) / 255
        arr = arr.reshape(-1, 1)
        sum = np.append(sum, arr)
    
    mean = np.mean(sum)
    std = np.std(sum)
    

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
