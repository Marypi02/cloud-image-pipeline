import os
import json
import requests

def download_and_split_coco_images(num_images):

    # 1. define the folders for the three categories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    dataset_dir = os.path.join(base_dir, "dataset")
    folder = ["small", "medium", "large"]

    # create the folders if they don't exist
    for f in folder:
        os.makedirs(os.path.join(dataset_dir, f), exist_ok=True)

    # 2. open the COCO annotation file
    annotation_dir = os.path.join(project_root, "coco_annotations", "annotations", "instances_val2017.json")
    
    try:
        with open(annotation_dir, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Annotation file not found.")
        return

    # 3. extract the list of images
    image_list = data.get("images", [])
    print(f"Total number of images: {len(image_list)}")

    downloaded_count = 0

    # 4. loop through the database and download the images
    for img in image_list:
        if downloaded_count >= num_images:
            break # stop since we have enough images for the test

        image_url = img.get("coco_url")
        file_name = img.get("file_name")

        if not image_url:
            print(f"Image URL not found for {file_name}. Skipping.")
            continue

        # check if the image already exists in any of the 3 folders
        image_already_exists = False
        for f in folder:
            folder_path = os.path.join(dataset_dir, f, file_name)
            if os.path.exists(folder_path):
                image_already_exists = True
                break # found the image in one of the folders, no need to check further

        if image_already_exists:
            print(f"Image already exists: {file_name}")
            downloaded_count += 1
            continue

        try:
            # download the pure image from the COCO server
            response = requests.get(image_url, timeout=10)

            if response.status_code == 200:
                pure_image_data = response.content

                # calculate the size of the image in kb
                size_kb = len(pure_image_data) / 1024

                # 5. save the image in the appropriate folder based on its size
                if size_kb < 150:
                    folder_name = "small"
                elif 150 <= size_kb < 350:
                    folder_name = "medium"
                else:
                    folder_name = "large"

                # save the image to the appropriate folder
                save_path = os.path.join(dataset_dir, folder_name, file_name)
                with open(save_path, "wb") as img_file:
                    img_file.write(pure_image_data)

                print(f"Downloaded: {file_name} | Size: {size_kb:.1f} KB | Folder: {folder_name}")
                downloaded_count += 1

        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
    
    print("\n" + "="*40)
    print("Download and split completed.")

    for f in folder:
        folder_path = os.path.join(dataset_dir, f)
        num_files = len(os.listdir(folder_path))
        print(f"Number of images in '{f}' folder: {num_files}")

if __name__ == "__main__":
    download_and_split_coco_images(200)