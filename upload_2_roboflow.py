from roboflow import Roboflow
import glob
import os
import time
import pandas as pd

# Initialize Roboflow client
rf = Roboflow(api_key="KEY")

# Directory paths for images and annotations
image_dir = "/content/output/"
label_dir = "/content/linha2-2/valid/labels/"
file_extensions = [".jpg", ".jpeg", ".png"]

# CSV file to track upload status
status_csv = "/content/upload_status.csv"

# Get the upload project from Roboflow workspace
print("Loading Roboflow workspace and project...")
upload_project = rf.workspace().project("linha2_warp")

# Function to handle the image upload with retry mechanism
def upload_with_retry(image_path, label_path, retries=3, delay=5):
    for attempt in range(retries):
        try:
            print(f"Attempting to upload {image_path} with annotation {label_path} (Attempt {attempt + 1})")
            upload_project.upload(image_path, label_path)
            print(f"Successfully uploaded {image_path}")
            return True
        except Exception as e:
            error_message = str(e)
            print(f"Error uploading {image_path}: {error_message}")
            # Handle specific error: Image was already annotated
            if "Image was already annotated" in error_message:
                print(f"Image {image_path} was already annotated. Marking as uploaded.")
                return True  # Consider as successfully uploaded
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to upload {image_path} after {retries} attempts.")
                return False

# Load or create the dataset for tracking upload status
def load_or_create_status_csv():
    # Create a new DataFrame with columns: 'file_name', 'uploaded'
    image_files = []
    for ext in file_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '*' + ext)))
    data = {'file_name': [os.path.basename(path) for path in image_files], 'uploaded': [0] * len(image_files)}
    df = pd.DataFrame(data)
    df.to_csv(status_csv, index=False)
    return df

# Update the CSV file after uploading
def update_status_csv(df):
    df.to_csv(status_csv, index=False)

# Main function to process uploads
def process_uploads():
    # Load or create the tracking DataFrame
    if os.path.exists(status_csv):
        df = pd.read_csv(status_csv)
    else:
        df = load_or_create_status_csv()

    try:
        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            if row['uploaded'] == 0:  # If not uploaded yet
                image_path = os.path.join(image_dir, row['file_name'])
                base_name = os.path.splitext(row['file_name'])[0]
                label_path = os.path.join(label_dir, base_name + '.txt')

                # Skip the upload if the annotation file is not found
                if os.path.exists(label_path):
                    success = upload_with_retry(image_path, label_path)
                    if success:
                        df.at[index, 'uploaded'] = 1  # Mark as uploaded
                        update_status_csv(df)  # Save progress
                    else:
                        print(f"Failed to upload {image_path}. Will retry next time.")
                else:
                    print(f"Warning: No annotation file found for {image_path}. Skipping upload.")
                    # Optionally, mark as skipped if annotation is missing
                    df.at[index, 'uploaded'] = -1  # Mark as skipped due to missing annotation
                    update_status_csv(df)
    except KeyboardInterrupt:
        print("\nUpload process interrupted by user. Saving progress...")
        update_status_csv(df)  # Save progress before exiting
        print("Progress saved. You can resume the upload later.")
        raise  # Re-raise the KeyboardInterrupt to exit the script

# Call the function to process uploads
process_uploads()
