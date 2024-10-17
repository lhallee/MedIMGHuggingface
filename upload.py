import os
import io
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
import argparse
from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm
from PIL import Image
from glob import glob


def main(args):
    test = args.test
    repo_id = args.repo_id
    image_extensions = args.image_extensions
    medical_image_extensions = args.medical_image_extensions
    parquet_file = args.parquet_file
    batch_size = args.batch_size
    counter = args.counter
    split = args.split
    image_file_paths, data = [], []

    if not test:
        api = HfApi()
        create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)

    # Remove existing parquet file if it exists
    if os.path.exists(parquet_file):
        os.remove(parquet_file)

    # Collect image file paths
    for folder in folders:
        folder_file_paths = []
        for dirpath, dirnames, filenames in tqdm(os.walk(folder)):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                # Check if the file has a known image extension
                if filename.lower().endswith(image_extensions + medical_image_extensions):
                    folder_file_paths.append(file_path)
                elif filename.lower().endswith(args.extensions_to_ignore):
                    # .raw are read by .mhd
                    continue
                else:
                    # Try to open files without a recognized extension to check if they are images
                    try:
                        with open(file_path, 'rb') as f:
                            img_bytes = f.read()
                        # Verify the image can be opened
                        Image.open(io.BytesIO(img_bytes))
                        folder_file_paths.append(file_path)
                        print(f"Found file without known extension: {file_path}")
                    except Exception as e:
                        pass
        print(f"Found {len(folder_file_paths)} image files in {folder}")
        image_file_paths.extend(folder_file_paths)
    print(f"Found {len(image_file_paths)} image files in total")


    # Process images in batches
    for image_path in tqdm(image_file_paths, desc='Reading'):
        try:
            if 'label' in image_path.lower():
                continue
            if image_path.lower().endswith('.dcm'):
                # Handle DICOM files separately
                try:
                    dcm_data = pydicom.dcmread(image_path)
                    image_array = dcm_data.pixel_array
                    normalized_image = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
                    pil_image = Image.fromarray(normalized_image)
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    file_name = f"{split}_" + image_path.split('medimg/')[-1].replace('\\', '/')
                    data.append({
                        'img_name': file_name,
                        'image': img_bytes,
                    })
                    counter += 1
                except Exception as e:
                    print(f"Error processing DICOM {image_path}: {e}")

            elif image_path.lower().endswith(medical_image_extensions):
                try:
                    # .mdh files read the .raw with GetArrayFromImage
                    image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
                    num_slices = image_array.shape[0]
                    if num_slices > 10:
                        indices = np.linspace(0, num_slices - 1, 10).astype(int)
                    else:
                        indices = np.arange(num_slices)
                    for index in indices:
                        slice_data = image_array[index, :, :]
                        normalized_slice = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
                        pil_image = Image.fromarray(normalized_slice)
                        img_byte_arr = io.BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        file_name = (
                            f"{split}_"
                            + image_path.split('medimg/')[-1].replace('\\', '/') + f'_slice{index}'
                        )
                        data.append({
                            'img_name': file_name,
                            'image': img_bytes,
                        })
                        counter += 1
                except Exception as e:
                    print(f"Error processing NIfTI {image_path}: {e}")

            elif image_path.lower().endswith(image_extensions):
                # Handle regular image files
                with open(image_path, 'rb') as f:
                    img_bytes = f.read()
                file_name = str(split) + '_' + image_path.split('medimg/')[-1].replace('\\', '/')
                data.append({
                    'img_name': file_name,
                    'image': img_bytes,
                })
                counter += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        # Save data in batches
        if counter > batch_size and not test:
            table = pa.Table.from_pandas(pd.DataFrame(data))
            pq.write_table(table, parquet_file, row_group_size=100)

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

    # Save any remaining data
    if counter > 0 and not test:
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


def get_args():
    parser = argparse.ArgumentParser(description='Upload images to Hugging Face Datasets')
    parser.add_argument('--repo_id', type=str, default='GleghornLab/MedIMG', help='Hugging Face repository ID')
    parser.add_argument('--data_location', type=str, default='E:/medimg/', help='Local filepath')
    parser.add_argument('--batch_size', type=int, default=10000, help='Number of images to upload in each batch')
    parser.add_argument('--base_path', type=str, default='E:/medimg', help='Base path to search for images')
    parser.add_argument('--split', type=int, default=1, help='Starting split number')
    parser.add_argument('--test', action='store_true', help='Test file parsing, no upload')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    folders = [ex.replace('\\', '/') for ex in glob(f'{args.data_location}*') if not ex.endswith('z01')]
    print(folders)
    args.extensions_to_ignore = ('.h5', '.raw', '.csv', '.tsv', '.txt', '.xlsx')
    args.image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.dcm', '.tif', '.ima')
    args.medical_image_extensions = ('.mhd', '.nii')
    
    args.parquet_file = 'images.parquet'
    args.counter = 0
    # login if needed
    # from huggingface_hub import login
    # login()

    main(args)