# ------------------------------------------------------------------------------------------------------
# DF_Face_Dict_csv_file_create.py
# Description       : Module which creates input csv file as an input to access images to apply XAI
# Author            : Nazneen Mansoor
# Date              : 01/07/2023
# -------------------------------------------------------------------------------------------------------

import os
import glob
import pandas as pd

from typing import Tuple, List

DATASETS_FOLDER = 'CelebAMask\images'

def generate_custom_dataset() -> Tuple[List[int], List[str]]:
    """

    Generate image id to extract corresponding labels from masked folder
    Args:
        None
    Return:
        img_ids: image id extracted from original images folder
        original_filepaths: image path of original images
    """
    files = []
    img_ids = []
    dataset_path = os.path.join(DATASETS_FOLDER)
    original_filepaths = glob.glob(f'{dataset_path}/*.jpg')

    # random_images = random.sample(original_filepaths, 4)

    for index in range(len(original_filepaths)):

        # getting the basename of each image
        files.append(os.path.basename(original_filepaths[index]))

        # extracting the image id after removing extension
        img_ids.append(os.path.splitext(files[index])[0])

    return img_ids, original_filepaths


def get_labels(
    image_ids: List[int],
    org_img_paths: List[str],
) -> None:
    """

    Fetch labels corresponding to each image id if present in the masked images folder
    Args:
        image_ids:
        org_img_paths:
    Return: None
    """
    dense_labels = []
    image_id_path = []
    masked_images_folder = 'CelebAMask\masked_images'

    for i in range(len(image_ids)):
        label_name = ''

        for file in os.listdir(masked_images_folder):

            if file.startswith(image_ids[i]):
                labels = os.path.join(masked_images_folder, file)
                label_name = f'{label_name}{labels};'


        if label_name != '':
            image_id_path.append(org_img_paths[i])
            dense_labels.append(label_name)
        # print(dense_labels)

    write_csv(image_id_path, dense_labels)
    read_update_csv()


def write_csv(
    image_id_path: List[str],
    dense_labels: List[str]
):
    """

    Creating csv file with original image paths and corresponding label paths
    Args:
        label_list:
        org_img_paths:
    Return: None
    """
    pd.DataFrame({'image': image_id_path, 'labels': dense_labels}).to_csv('DF_Face_Dict_Attributes.csv', index=False)
    print('row created')


def read_update_csv(
):

    df = pd.read_csv('DF_Face_Dict_Attributes.csv')
    df["labels"] = df["labels"].str[:-1]
    df.to_csv('DF_Face_Dict_Attributes.csv', index=None, header=True)


if __name__ == '__main__':
    image_ids, org_img_paths = generate_custom_dataset()
    get_labels(image_ids, org_img_paths)



