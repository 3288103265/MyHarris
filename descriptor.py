import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from PIL import Image

al1 = np.array(Image.open('truck1.jpg').convert('L')).T
al2 = np.array(Image.open('truck2.jpg').convert('L')).T

def harris(im, sigma=3):
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet-0.05*(Wtr**2)


def harrisPoints(harrisim, min_dist=15, threshold=0.1):
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    coords = np.array(harrisim_t.nonzero()).T

    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    index = np.argsort(candidate_values)[::-1]

    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):

    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()


def get_descriptors(image, filtered_coords, wid=5):  # return pixel value

    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - wid:coords[0] + wid + 1,
                coords[1] - wid:coords[1] + wid + 1].flatten()
        desc.append(patch)

    return desc


def match(desc1, desc2, threshold=0.5):

    n = len(desc1[0])

    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores


def match_twosided(desc1, desc2, threshold=0.5):  # Same as above, two sided symmetric version

    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    ndx_12 = np.where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12


def appendimages(im1, im2):  # the appended images displayed side by side for image mapping

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    # if elseif statement
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)

    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):  # Plots the matches between both images

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plt.plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'm', linewidth =0.75, linestyle='--')
    plt.axis('off')
    plt.savefig('result.png', dpi=600)


wid = 5
harrisim1 = harris(al1, 3)
filtered_coords1 = harrisPoints(harrisim1, wid + 1)
d1 = get_descriptors(al1, filtered_coords1, wid)
harrisim2 = harris(al2, 3)
filtered_coords2 = harrisPoints(harrisim2, wid + 1)
d2 = get_descriptors(al2, filtered_coords2, wid)

matches = match_twosided(d1, d2)

plt.figure()
plt.gray()
plot_matches(al1, al2, filtered_coords1, filtered_coords2, matches)
plt.show()
# cv2.imshow('img1',al1)
# cv2.waitKey()
# cv2.destroyAllWindows()
