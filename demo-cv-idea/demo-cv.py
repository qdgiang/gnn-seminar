import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
# from scipy.spatial import Delaunay
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic


def run(file_name: str, no_segments: int, compact: float):
    img_fn = file_name + '.jpg'
    new_img = cv.imread(img_fn)
    img = new_img.astype(np.float32) / 255.

    # SLIC
    segments = slic(img, n_segments=no_segments, compactness=compact)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    plt.imshow(mark_boundaries(img, segments), cmap='gray')
    plt.scatter(centers[:, 1], centers[:, 0], c='y')

    for i in range(bneighbors.shape[1]):
        y0, x0 = centers[bneighbors[0, i] - 1]
        y1, x1 = centers[bneighbors[1, i] - 1]

        line = Line2D([x0, x1], [y0, y1], alpha=1)
        ax.add_line(line)
    print("Index list: ", segments_ids)
    print("Edge list:\n", bneighbors)
    plt.show()


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
