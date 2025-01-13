# multiple CPU, but copy the image to output if donot find the colorchecker.

import os
import glob
import numpy as np
import csv
import colour
from scipy.spatial.distance import cdist
from colour_checker_detection import detect_colour_checkers_segmentation
import multiprocessing as mp
import shutil  # Import shutil to copy files

input_folder = '/content/linha2-2/valid/images'
output_folder = '/content/output/'
csv_file_path = os.path.join(output_folder, 'delta_e_results.csv')

def tps_3d_warp(image, source_points, destination_points):
    """
    Applies TPS-3D warping to calibrate the image colors.
    :param image: Input image (H x W x 3).
    :param source_points: Source RGB values from the image (N x 3).
    :param destination_points: Reference RGB values (N x 3).
    :return: Color calibrated image.
    """
    N = source_points.shape[0]

    # Build matrix P (E2)
    ones = np.ones((N, 1))
    P = np.hstack((ones, source_points))  # N x 4

    # Build matrix K (E3 and E4)
    r = cdist(source_points, source_points, 'euclidean')  # N x N
    # Avoid log(0)
    np.fill_diagonal(r, 1e-20)
    U = 2 * (r ** 2) * np.log(r)
    K = U  # N x N

    # Assemble the L matrix (E1)
    O = np.zeros((4, 4))
    L_upper = np.hstack((K, P))
    L_lower = np.hstack((P.T, O))
    L = np.vstack((L_upper, L_lower))  # (N + 4) x (N + 4)

    # Build V matrix (E5)
    V = np.vstack((destination_points, np.zeros((4, 3))))  # (N + 4) x 3

    # Solve for the transformation parameters [W; A]
    params = np.linalg.solve(L, V)
    W = params[:N, :]  # N x 3
    A = params[N:, :]  # 4 x 3

    # Apply the transformation to each pixel in the image
    H, W_img, _ = image.shape
    image_reshaped = image.reshape(-1, 3)  # (H * W) x 3

    # Build matrix P' for all pixels
    ones_all = np.ones((image_reshaped.shape[0], 1))
    P_all = np.hstack((ones_all, image_reshaped))  # (H * W) x 4

    # Compute U for all pixels
    r_all = cdist(image_reshaped, source_points, 'euclidean')  # (H * W) x N
    # Avoid log(0)
    r_all[r_all == 0] = 1e-20
    U_all = 2 * (r_all ** 2) * np.log(r_all)  # (H * W) x N

    # Compute the transformed RGB values
    transformed = U_all @ W + P_all @ A  # (H * W) x 3

    # Reshape back to image dimensions
    transformed_image = transformed.reshape(H, W_img, 3)

    # Clip values to valid range
    transformed_image = np.clip(transformed_image, 0, 1)

    return transformed_image

def process_image(image_path):
    """Process a single image and return its results."""
    RGB = colour.io.read_image(image_path)
    image = colour.cctf_decoding(RGB)
    print ("input:", image_path)
    output_path = os.path.join(output_folder, os.path.basename(image_path))

    detected = detect_colour_checkers_segmentation(image, additional_data=True)

    if not detected:
        print(f"No colour checker detected in the image {image_path}. Copying image as is.")
        # Copy the image to the output folder
        shutil.copy(image_path, output_path)
        return [os.path.basename(image_path), None, None]

    D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS[
        'ColorChecker24 - Before November 2014']

    REFERENCE_SWATCHES = colour.XYZ_to_RGB(
        colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
        REFERENCE_COLOUR_CHECKER.illuminant, D65,
        colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)

    REFERENCE_SWATCHES = np.clip(REFERENCE_SWATCHES, 0, 1)

    for data in detected:
        swatches = data.swatch_colours  # N x 3
        swatches = np.clip(swatches, 0, 1)

        # Perform TPS-3D warping
        calibrated_image = tps_3d_warp(image, swatches, REFERENCE_SWATCHES)
        colour.io.write_image(colour.cctf_encoding(calibrated_image), output_path)

        reais = colour.XYZ_to_Lab(colour.RGB_to_XYZ(
            swatches, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))
        cc = colour.XYZ_to_Lab(colour.RGB_to_XYZ(
            REFERENCE_SWATCHES, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))
        delta_E_before = colour.difference.delta_E_CIE2000(reais, cc)
        delta_E_before_sum = np.sum(delta_E_before)

        swatches_f = tps_3d_warp(swatches.reshape(1, -1, 3), swatches, REFERENCE_SWATCHES)
        swatches_f = swatches_f.reshape(-1, 3)

        corr = colour.XYZ_to_Lab(colour.RGB_to_XYZ(
            swatches_f, D65, D65, colour.RGB_COLOURSPACES['sRGB'].matrix_RGB_to_XYZ))
        delta_E_after = colour.difference.delta_E_CIE2000(corr, cc)
        delta_E_after_sum = np.sum(delta_E_after)

        print ("Output:", os.path.basename(image_path))
        return [os.path.basename(image_path), delta_E_before_sum, delta_E_after_sum]

def main():
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_folder, '*.[jJpP][pPnN][gG]*'))

    # Create a multiprocessing Pool with 6 processes
    with mp.Pool(processes=6) as pool:
        results = pool.map(process_image, image_paths)

    # Write the results to the CSV file
    csv_data = [["file_name", "delta_e_before", "delta_e_after"]] + results
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"CSV file saved at {csv_file_path}")

if __name__ == '__main__':
    main()
