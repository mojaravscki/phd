import os
import glob
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
from helpers_opencv import process_annotation_file

def process_dataset(images_dir, labels_dir, sizes, output_csv, color_mode="RGB"):
    # Ensure the output directory exists (where CSV will be saved)
    output_dir = os.path.dirname(output_csv)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory {output_dir} ensured.")
    
    # Define a directory for cropped images (sub-folder)
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)
    print(f"Cropped images will be saved to {cropped_dir}.")
    
    # Find annotation files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not label_files:
        print(f"No annotation files found in {labels_dir}. Skipping...")
        return
    
    print(f"Found {len(label_files)} annotation files.")
    
    # Prepare partial function to inject fixed arguments for parallel calls
    func = partial(
        process_annotation_file,
        images_dir=images_dir,
        sizes=sizes,
        color_mode=color_mode,
        output_dir=cropped_dir
    )
    
    # Create the CSV file and write the header
    with open(output_csv, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ["image_name", "class_number"]
        
        # Add channel-specific headers based on color mode
        if color_mode in ["RGB", "LAB"]:
            # For both RGB and LAB, we have 3 channels
            channels = ['R', 'G', 'B'] if color_mode == "RGB" else ['L', 'A', 'B']
            header.extend(f"{ch}_{i}" for ch in channels for i in range(sizes * sizes))
        elif color_mode == "H":
            # For Hue only, we have 1 channel
            header.extend(f"H_{i}" for i in range(sizes * sizes))
        
        csv_writer.writerow(header)
        
        # Use parallel processing (up to 8 workers or # of cores)
        max_workers = min(8, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # executor.map returns an iterator of results
            for idx, result in enumerate(executor.map(func, label_files, chunksize=10), 1):
                if result:
                    csv_writer.writerows(result)
                if idx % 100 == 0:
                    print(f"Processed {idx} annotation files...")
    
    print(f"Extraction completed for {color_mode}. CSV saved to: {output_csv}")


if __name__ == "__main__":
    # On some platforms, especially Windows or macOS in certain cases,
    # 'spawn' can be safer when starting processes.
    multiprocessing.set_start_method("spawn")
    
    # Configuration
    pathsname = "aug" ## aug or no_aug
    hhhh = "h1"
    names = ["test", "valid", "train"]
    folders = ["h1_asis", "h2_warp", "h3_ahe", "h7_combined"] ## USED FOR no_aug
    #["h1_asis", "h2_warp", "h3_ahe"] ## USED FOR aug
    sizes = 32
    color_mode = "LAB"  # Could also be "LAB" or "H"

    # Loop through each dataset partition
    for folder in folders:
        for name in names:
            print(f"Processing dataset: {name}")
            # Update these paths to your actual dataset directories

            #             "/Users/macbookpro/Documents/Projects/phd/roboflow/aug/h1_asis/train"
            images_dir = f"/Users/macbookpro/Documents/Projects/phd/roboflow/{pathsname}/{folder}/{name}/images"
            labels_dir = f"/Users/macbookpro/Documents/Projects/phd/roboflow/{pathsname}/{folder}/{name}/labels"
            output_csv = f"/Users/macbookpro/Documents/Projects/phd/roboflow/output/{pathsname}/{folder}/{folder}_{name}_{pathsname}_{color_mode}_{sizes}x{sizes}.csv"
            #             "/Users/macbookpro/Documents/Projects/phd/roboflow/output/no_aug/h1_asis"
            
            process_dataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                sizes=sizes,
                output_csv=output_csv,
                color_mode=color_mode
            )
        
        print(f"Finished processing dataset: {name}")


### H1 No Agumentation

#H1_h1 - AS IS

#H1_h2 - Warp

#H1_h3 - AHE




### H2 Augmentation

#H2_h1 - AS IS

#H2_h2 - Warp

#H2_h3 - AHE



