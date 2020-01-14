"""
This module is a collection of functions to segment macroscopic bacterial colonies
growing in Petri dishes
"""
# Author: Guillaume Witz, Science IT Support, Bern University, 2019
# License: BSD3

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import skimage.segmentation
import skimage.feature
import skimage.io
from skimage.transform import hough_line, hough_line_peaks

import scipy
from scipy.ndimage import uniform_filter
import scipy.ndimage as ndi

from pyefd import (
    elliptic_fourier_descriptors,
    reconstruct_contour,
    calculate_dc_coefficients,
)

import matplotlib

cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))


def find_dish(image):
    """Find the bounding box of the petri dish
    
    Parameters
    ----------
    image : 2D numpy array
        image of petri dish with colony
        
    Returns
    -------
    im_cut : 2D numpy array 
        cropped image around the petri dish
    mask_cut : 2D numpy array
        mask of the petry dish (disk-like)
    box : numpy array
        bounding box coordinates used for cropping
    """

    petri_lab = skimage.morphology.label(image > 10)
    petri_lab = petri_lab == petri_lab[int(image.shape[0] / 2), int(image.shape[1] / 2)]
    petri_lab = skimage.morphology.binary_closing(petri_lab, np.ones((5, 5)))

    frac = 0.85
    im_reg = skimage.measure.regionprops(petri_lab.astype(int))

    im_cut = image[
        im_reg[0].bbox[0] : im_reg[0].bbox[2], im_reg[0].bbox[1] : im_reg[0].bbox[3]
    ]
    mask_cut = im_reg[0].image

    box = im_reg[0].bbox

    return im_cut, mask_cut, box


def remove_spots(image, lim=50):
    """Remove bright spots and replace them with the average
    value of the surrounding region
    
    Parameters
    ----------
    image : 2D numpy array
        image of petri dish with colony
    lim : float
        intensity threshold
        
    Returns
    -------
    im_clean : 2D numpy array 
        cleaned image
    """

    hole_size = 21

    # create ring-shaped filter
    X, Y = np.meshgrid(np.arange(hole_size), np.arange(hole_size))
    hole_filt = (
        ((X - (hole_size - 1) / 2) ** 2 + (Y - (hole_size - 1) / 2) ** 2) ** 0.5 < 10
    ) & (((X - (hole_size - 1) / 2) ** 2 + (Y - (hole_size - 1) / 2) ** 2) ** 0.5 > 5)

    # mean filter the image with the ring filter
    im_hole = skimage.filters.rank.mean(image, hole_filt)

    # bottom hat filter to find spots
    # im_bottom = skimage.filters.rank.bottomhat(image, skimage.morphology.disk(4))
    im_bottom = skimage.morphology.white_tophat(image, skimage.morphology.disk(5))
    im_bottom_th = im_bottom > lim

    # replace spots by ring-averaged values
    im_clean = image.copy()
    im_clean[im_bottom_th] = im_hole[im_bottom_th]

    return im_clean


def rough_colony_farid(image):
    """Find a rough localization of the colony by detecting its
    contour using the Farid filter
    
    Parameters
    ----------
    image : 2D numpy array
        image of petri dish with colony
    
    Returns
    -------
    new_mask : 2D numpy array 
        rough mask of the colony
    """

    # recover regions with borders by thresholding a Farid filtered image
    farid = skimage.filters.farid(skimage.filters.gaussian(image, 2)) > 0.002

    # remove straight lines (I guess marks on dish) using hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(farid, theta=tested_angles)

    large_gauss = skimage.filters.gaussian(image, 5, preserve_range=True)

    # each line appears as a doublet after Farid filtering
    # we suppress here two doublets in the image by taking the four largest peaks
    # if an image doesn't have lines, this operation is not very dammaging but we
    # could set a threshold in the future
    # we replace line regions with gaussian filtered values
    origin = np.array((0, farid.shape[1]))
    for accum, angle, dist in zip(
        *hough_line_peaks(h, theta, d, min_distance=1, num_peaks=4)
    ):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        line_coord = skimage.draw.line(
            int(y0), int(origin[0]), int(y1), int(origin[1]) - 1
        )
        empty = np.zeros(farid.shape)
        line = np.stack((line_coord[0], line_coord[1]), axis=1)
        line = line[line[:, 1] < empty.shape[1], :]
        line = line[line[:, 0] < empty.shape[0], :]
        line = line[line[:, 1] > 0, :]
        line = line[line[:, 0] > 0, :]
        empty[line[:, 0], line[:, 1]] = 1
        empty_dil = skimage.morphology.binary_dilation(
            empty, skimage.morphology.disk(5)
        )
        image[empty_dil] = large_gauss[empty_dil]

    # farid filer again on the image without lines and clear border
    farid = skimage.filters.farid(skimage.filters.gaussian(image, 2)) > 0.002
    farid2 = (
        skimage.segmentation.clear_border(
            skimage.morphology.label(farid), buffer_size=20
        )
        > 0
    )

    # remove small objects and close
    farid_lab = skimage.morphology.label(farid2)
    farid_reg = skimage.measure.regionprops(farid_lab)

    farid_indices = np.array(
        [0] + [x.label if x.area > 500 else 0 for x in farid_reg]
    ).astype(int)
    farid3 = farid_indices[farid_lab] > 0
    farid4 = skimage.morphology.binary_closing(farid3, skimage.morphology.disk(3))

    # sort remaining regions by filled_area and remove objects only present in the sourrounding
    # area by removing objects that have no pixel within 200 px of the center
    farid5 = skimage.morphology.label(farid4)
    reg = skimage.measure.regionprops_table(
        farid5, properties=("coords", "label", "filled_area", "filled_image", "bbox")
    )
    reg_pd = pd.DataFrame(reg)
    reg_pd["min_dist"] = [
        np.min(
            (
                (reg["coords"][x][:, 0] - farid3.shape[0] / 2) ** 2
                + (reg["coords"][x][:, 1] - farid3.shape[1] / 2) ** 2
            )
            ** 0.5
        )
        for x in range(len(reg["coords"]))
    ]
    reg_pd = (reg_pd[reg_pd.min_dist < 200]).sort_values(
        by="filled_area", ascending=False
    )
    new_mask = np.zeros(farid3.shape)
    new_mask[
        reg_pd.iloc[0]["bbox-0"] : reg_pd.iloc[0]["bbox-2"],
        reg_pd.iloc[0]["bbox-1"] : reg_pd.iloc[0]["bbox-3"],
    ] = reg_pd.iloc[0].filled_image

    return new_mask


def create_contour(mask):
    """Turn as mask into a contour polygon. The polygon is smoothed
    via Fourier descriptors
    
    Parameters
    ----------
    mask : 2D numpy array
        mask of a region
    
    Returns
    -------
    recon : Nx2 numpy array 
        coordinates of contour
    """

    # extract contour by recovering contour of filled mask
    reg = skimage.measure.regionprops_table(
        skimage.morphology.label(mask),
        properties=(
            "label",
            "area",
            "filled_area",
            "eccentricity",
            "extent",
            "filled_image",
            "coords",
            "bbox",
        ),
    )
    regp = pd.DataFrame(reg).sort_values(by="filled_area", ascending=False)

    new_mask2 = np.zeros(mask.shape)
    new_mask2[
        regp.iloc[0]["bbox-0"] : regp.iloc[0]["bbox-2"],
        regp.iloc[0]["bbox-1"] : regp.iloc[0]["bbox-3"],
    ] = regp.iloc[0].filled_image
    area = regp.iloc[0].filled_area

    contour = skimage.measure.find_contours(new_mask2, 0.8)

    sel_contour = contour[np.argmax([len(x) for x in contour])]

    # calculate a smoothed verions using a reconstruction via Fourier descriptors
    coeffs = elliptic_fourier_descriptors(sel_contour, order=100, normalize=False)
    coef0 = calculate_dc_coefficients(sel_contour)
    recon = reconstruct_contour(coeffs, locus=coef0, num_points=1500)

    return recon, area


def calculate_background(image, center_mask):
    """Calculate a background image by interpolating the region 
    occupied by the colony
    
    Parameters
    ----------
    image : 2D numpy array
        image for which to calculate background
    center_mask : 2D numpy array
        mask of the region to interpolate
    
    Returns
    -------
    rescaled_back : 2D numpy array 
        interpolated background image
    """

    # General principle: we want to replace the mask region by interpolated regions based
    # on sourrounding values. To ensure this is smooth, the border region needs to be smoothed
    # but since the mask region contains nan's we have to implement a manual local average taking¨
    # into account nan's. For that we split the image into 10x10 region and do a nanmean() on each of
    # them. To be able to split the image, it is padded with the appropriate number of columns/rows

    # dilate the region to interpolate
    center_mask = skimage.morphology.binary_dilation(
        center_mask, skimage.morphology.disk(20)
    )

    # rescale the image by 2 to make computing faster and split image into 10x10 patches
    patch = 10
    patch2 = int(patch / 2)
    image_small = image[::2, ::2]
    mask_small = center_mask[::2, ::2]
    topad = patch - np.array(image_small.shape) % patch

    # mask image and pad it to ensure that one can split it into 10x10 region and that the border
    # regions are not lost
    im_pad = np.pad(
        image_small * (1 - mask_small),
        ((patch2, topad[0] + patch2), (patch2, topad[1] + patch2)),
        mode="constant",
    )
    im_pad = im_pad.astype(float)
    im_pad[im_pad == 0] = np.nan

    # split and nanmean() local averaging
    split = skimage.util.view_as_windows(im_pad, 10)
    local_mean = np.nanmean(np.nanmean(split, axis=2), axis=2)

    # pad the image and maks so that they have the same size as the local nanmean() matrix
    image_pad = np.pad(
        image_small, ((0, topad[0] + 1), (0, topad[1] + 1)), mode="constant"
    )
    mask_pad = np.pad(
        mask_small, ((0, topad[0] + 1), (0, topad[1] + 1)), mode="constant"
    )

    # the region directly sourrounding the masks has an average of only few pixels (lots of nan's)
    # and is therefore noisy. We remove it and replace it with nan's
    local_mean[
        skimage.morphology.binary_dilation(mask_pad, skimage.morphology.disk(3))
    ] = np.nan

    ##Now we interpolate the nan region linearly
    x = np.arange(0, local_mean.shape[1])
    y = np.arange(0, local_mean.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(local_mean)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    interp = scipy.interpolate.griddata(
        (x1, y1), newarr.ravel(), (xx, yy), method="linear"
    )

    # finally we rescale to original size
    rescaled_back = skimage.transform.resize(interp, image.shape)

    return rescaled_back


def correct_background(image, background):
    """Correct the background of an image using a calculated 
    interpolated background
    
    Parameters
    ----------
    image : 2D numpy array
        image to correct
    background : 2D numpy array
        background image
    
    Returns
    -------
    im_enhnaced : 2D numpy array 
        corrected and enhanced image
    """
    corr_im = uniform_filter(image - background, 4)
    corr_im = 255 * (corr_im - corr_im.min()) / (np.max(corr_im) - np.min(corr_im))
    corr_im = corr_im.astype(int)
    im_enhanced = skimage.filters.rank.enhance_contrast(
        corr_im, skimage.morphology.disk(3)
    )
    return im_enhanced


def find_branches(contour, prominence = 20, max_peak_width = 100):
    """Find branches in the colony by detecting peaks in the contour 
    distance from the center of mass of the colony
    
    Parameters
    ----------
    contour : Nx2 numpy array
        contour coordinates
    
    Returns
    -------
    peak :  numpy array 
        peak indices in contour
    """
    # If the start of the contour is a peak, it's not picked up by the find_peaks()
    # function. So we replicate part of the contour
    cm = np.mean(contour, axis=0)
    center_dist = ((contour[:, 0] - cm[0]) ** 2 + (contour[:, 1] - cm[1]) ** 2) ** 0.5
    center_dist = np.concatenate((center_dist, center_dist[1:200]))
    peak, peak_info = scipy.signal.find_peaks(center_dist, distance=10, width=7,prominence=prominence)
    peak[(peak_info["prominences"] > prominence) & (peak_info["widths"] < max_peak_width)]
    peak = peak[peak < len(contour) + 1] % len(contour)

    return peak


def complete_analysis(file):
    try:
        image = skimage.io.imread(file)[:, :, 1]
        image_scaled = image[::2, ::2]
        image_scaled_crop, mask_scaled_crop, bbox = find_dish(image_scaled)
        image_crop = image[2 * bbox[0] : 2 * bbox[2], 2 * bbox[1] : 2 * bbox[3]]

        # remove spots
        image_scaled_clean = remove_spots(image_scaled_crop)

        # find rough contour of colony
        rough_colony = rough_colony_farid(image_scaled_clean)
        rough_colony_large = skimage.morphology.binary_dilation(
            rough_colony, skimage.morphology.disk(20)
        )

        # calculate backgorund by interpolating the colony area
        back = calculate_background(image_scaled_clean, rough_colony)
        back[np.isnan(back)] = 0

        # correct background of image and enhance contrast
        im_enhanced = correct_background(image_scaled_clean, back)

        # find an Otsu threshold by observing a band around the rough colony
        threshold = skimage.filters.threshold_otsu(
            im_enhanced[
                rough_colony_large
                ^ skimage.morphology.binary_erosion(
                    rough_colony, skimage.morphology.disk(10)
                )
            ]
        )

        # final clean-up
        final_colony = skimage.morphology.binary_closing(
            (rough_colony_large * im_enhanced) > threshold, skimage.morphology.disk(1)
        )  # *plate_scaled_mask

        # create contour
        contour, area = create_contour(final_colony)

        # find peaks
        peak = find_branches(contour)

        contour_upscaled = 2*(contour + np.array([bbox[0],bbox[1]]))

        #normalize area by dish area
        area = area / np.sum(mask_scaled_crop)

        #find center of mass of plate
        center_mass = 2*(np.array(ndi.measurements.center_of_mass(mask_scaled_crop))+ np.array([bbox[0],bbox[1]]))        

        return contour_upscaled, peak, area, center_mass
    
    except:
        return None, None, None, None