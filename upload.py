import os
from glob import glob
import SimpleITK as sitk
import io
import numpy as np
from PIL import Image
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm


repo_id = 'GleghornLab/MedIMG'
create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
folders = [ex.replace('\\', '/') for ex in glob('E:/medimg/*') if not ex.endswith('z01')]
print(folders)
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
medical_image_extensions = ('.mhd',)
api = HfApi()
batch_size = 10000
image_file_paths = []
parquet_file = 'images.parquet'
data = []
split, counter = 1, 0 # change split based on where starting


# Remove existing parquet file if it exists
if os.path.exists(parquet_file):
    os.remove(parquet_file)


# Collect image file paths
for folder in folders:
    folder_file_paths = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(image_extensions + medical_image_extensions):
                file_path = os.path.join(dirpath, filename)
                folder_file_paths.append(file_path)
    print(f"Found {len(folder_file_paths)} image files in {folder}")
    image_file_paths.extend(folder_file_paths)
print(f"Found {len(image_file_paths)} image files in total")


# Process images in batches
for image_path in tqdm(image_file_paths, desc='Reading'):
    try:
        if image_path.lower().endswith(medical_image_extensions):
            image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            num_slices = image_array.shape[0]
            if num_slices > 10: # Generate 10 evenly spaced indices
                indices = np.linspace(0, num_slices - 1, 10).astype(int)
            else: # Use all the slices
                indices = np.arange(num_slices)
            for i, index in enumerate(indices):
                # Extract the slice
                slice_data = image_array[index, :, :]
                # Convert to PIL Image
                pil_image = Image.fromarray(slice_data)
                # Save to bytes
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')  # Save the image to the byte array
                img_bytes = img_byte_arr.getvalue()
                # Create a file name
                file_name = (
                    f"{split}_"
                    + image_path.split('medimg/')[-1].replace('\\', '/').replace('.mhd', f'_slice{index}.png')
                )
                data.append({
                    'img_name': file_name,
                    'image': img_bytes,
                })
                counter += 1  # Increment counter by 1 for each slice
        else:
            # Existing code for other images
            with open(image_path, 'rb') as f:
                img_bytes = f.read()
            # Verify the image can be opened
            image = Image.open(io.BytesIO(img_bytes))
            file_name = str(split) + '_' + image_path.split('medimg/')[-1].replace('\\', '/')
            data.append({
                'img_name': file_name,
                'image': img_bytes,
            })
            counter += 1
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

    if counter > batch_size:
        table = pa.Table.from_pandas(pd.DataFrame(data))
        pq.write_table(table, parquet_file)

        api.upload_file(
            path_or_fileobj=parquet_file,
            path_in_repo=f"data/{split}/{parquet_file}",
            repo_id=repo_id,
            repo_type='dataset',
        )
        print(f"Batch {split} uploaded to https://huggingface.co/datasets/{repo_id}/data/{split}/{parquet_file}")

        # Clean up
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        data = []
        split += 1
        counter = 0

if counter > 0:
    table = pa.Table.from_pandas(pd.DataFrame(data))
    pq.write_table(table, parquet_file)

    api.upload_file(
        path_or_fileobj=parquet_file,
        path_in_repo=f"data/{split}/{parquet_file}",
        repo_id=repo_id,
        repo_type='dataset',
    )
    print(f"Batch {split} uploaded to https://huggingface.co/datasets/{repo_id}")

    # Clean up
    if os.path.exists(parquet_file):
        os.remove(parquet_file)
