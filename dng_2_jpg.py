import os
import argparse
import rawpy
import imageio

def convert_dng_to_jpg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.dng'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_folder, output_filename)

            print(f'Converting {filename} to {output_filename}...')

            # Read the DNG file
            with rawpy.imread(input_path) as raw:
                rgb_image = raw.postprocess()

            # Save the image as JPG
            imageio.imsave(output_path, rgb_image)

    print('Conversion completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DNG images to JPG.')
    parser.add_argument('--input_folder', required=True, help='Folder containing DNG images.')
    parser.add_argument('--output_folder', required=True, help='Folder to save JPG images.')
    args = parser.parse_args()

    convert_dng_to_jpg(args.input_folder, args.output_folder)
