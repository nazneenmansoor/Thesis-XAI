# -----------------------------------------------------------------------------------------------------------------------------------------------------
# visual_dict_dataloader.py
# Description       : Module which loads input data and labelled concepts are given
# Author            : Nazneen Mansoor
# Date              : 15/12/2022
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import concurrent.futures
import math
from typing import List, Dict

import torch
import csv
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

from settings import get_settings

opts = get_settings()
global subcat

class VisDict():
    """
    Reads the input data and their corresponding labels
    from the image paths specified in the DF_Face_Dict_Attributes csv file.
    Labels are mapped with the dense labels specified in the DeepFake_Face_dictionary dictionary
    which are a set of spatial features of face.
    """

    def __init__(
        self,
        root,
        batch_size: int,
    ) -> None:

        """Reading the csv file for getting the image path

        Args:
            root:
            batch_size: batch size specified in settings module
        Return: None
        """

        self.images = []
        self.batch_size = batch_size
        # Multitask CNN for face and facial landmark detection on images
        self.mtcnn = MTCNN(image_size=256)

        # input csv file with image id and labels
        files = ['DF_Face_Dict_Attributes.csv']
        # file handling operations
        for file in files:
            with open(root + file, 'r') as f:
                reader = csv.DictReader(f)  # access csv data using indexes
                for line in reader:
                    self.images.append(line)

        self.num_batches = math.ceil(len(self.images) / self.batch_size)  # compute number of batches
        self.dense_label_mapping = dict((x, []) for x in opts.dense_labels)
        self.queue = []
        self.images_per_label()

    def get_sample(
        self,
        idx: torch.Tensor,
        crop: bool,
    ) -> Dict[List[str], List[str]]:
        """Get current sample images and labels

        Args:
            idx:  image id
            crop: face crop variable, either True or False
        Return:
            sample: dictionary with image and labels
        """

        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = self.images[idx]['image']  # path to access images
        img = Image.open('DeepFake_Face_dictionary/' + img_path)  # opening images
        # img = Image.open(img_path)  # opening images
        img = img.resize((opts.input_size, opts.input_size))  # resizing the images

        if img.mode == 'L':
            rgb_img = Image.new('RGB', img.size)  # creating a new image object with RGB mode
            rgb_img.paste(img)
            img = rgb_img

        sample = {}
        tr_tensor = transforms.ToTensor()  # variable to convert into tensor
        tr_PIL = transforms.ToPILImage()  # converts to a PIL image

        # convert images to tensor
        if crop:
            cropped = self.mtcnn(img.copy())
            crop_points = self.mtcnn.detect(img.copy(), landmarks=False)
            cp = crop_points[0]
            NoneType = type(None)
            if isinstance(cp, NoneType):
                cp = np.array([0, 0, opts.input_size, opts.input_size])
                img = img.resize((opts.input_size, opts.input_size))
                img = tr_tensor(img)
            else:
                cp = cp[0]
                img = cropped
                img = tr_PIL(img)
                img = img.resize((opts.input_size, opts.input_size))
                img = tr_tensor(img)
        else:
            img = tr_tensor(img)

        # setting values into sample dict
        sample['image'] = img
        sample['i'] = idx
        sample['labels'] = {}
        sample['fn'] = img_path
        curr_labels = self.images[idx]['labels'].split(';')

        # loop through each image label path
        for path in curr_labels:

            label = Image.open('DeepFake_Face_dictionary/' + path).convert('L')  # open the images labelled with spatial concepts


            for dl in opts.dense_labels:
                if dl in path:
                    subcat = dl
            label = label.resize((opts.input_size, opts.input_size), Image.NEAREST)

            if crop:
                label = label.crop((cp[0], cp[1], cp[2], cp[3]))
                label = label.resize((opts.input_size, opts.input_size), Image.NEAREST)

            label = tr_tensor(label)
            sample['labels'][subcat] = label

        return sample


    def get_image(
        self,
        idx: torch.Tensor,
        crop: bool,
    ) -> torch.Tensor:

        """Get current images

        Args:
            idx: image id
            crop: face crop variable, either True or False
        Return:
            img: images specified in the path
        """

        if torch.is_tensor(idx):
            idx = idx.to_list()

        tr_tensor = transforms.ToTensor()  # variable to convert into tensor
        tr_PIL = transforms.ToPILImage()  # converts to a PIL image

        img_path = self.images[idx]['image']
        img = Image.open('DeepFake_Face_dictionary/' + img_path)
        # img = Image.open(img_path)
        img = img.resize((opts.input_size, opts.input_size))

        if img.mode == 'L':
            rgb_img = Image.new('RGB', img.size)  # creating a new image object with RGB mode
            rgb_img.paste(img)
            img = rgb_img

        # check if face crop is True or False
        if crop:
            cropped = self.mtcnn(img.copy())
            crop_points = self.mtcnn.detect(img.copy(), landmarks=False)
            cp = crop_points[0]
            NoneType = type(None)
            if isinstance(cp, NoneType):
                img = img.resize((opts.input_size, opts.input_size))
                img = tr_tensor(img)
            else:
                img = cropped
                img = tr_PIL(img)
                img = img.resize((opts.input_size, opts.input_size))
                img = tr_tensor(img)

        else:
            img = tr_tensor(img)

        return img

    # @threadsafe_generator
    def get_batches(
        self,
        only_image= False,
        face_crop= False,
    ):

        """Get batch

        Args:
            face_crop: face crop variable, either True or Falseself.loader.get_batches(only_image=True, face_crop=face_crop)
            only_image: image only or with labels, True or False
        Return:
            batch:
        """

        num_images = len(self.images)  # number of images as input data

        for idx in range(0, num_images, self.batch_size):
            batch = []
            if face_crop:
                for j in range(idx, min(idx + self.batch_size, num_images)):

                    if only_image:
                        # invoking get_image with face_crop = False
                        curr_sample = self.get_image(j, face_crop)
                    else:
                        # invoking get_sample with face_crop = False
                        curr_sample = self.get_sample(j, face_crop)
                    batch.append(curr_sample)
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    indexes = range(idx, min(idx + self.batch_size, num_images))
                    crops = [face_crop] * len(indexes)
                    if only_image:
                        results = executor.map(self.get_image, indexes, crops)
                    else:
                        results = executor.map(self.get_sample, indexes, crops)
                    for res in results:
                        batch.append(res)

            yield batch


    def images_per_label(
        self,
    ) -> None:
        """Get the labels for each image
        Return: None
        """
        print('Mapping labels to images')

        # loop through the number of images
        for idx in range(len(self.images)):
            labels = self.images[idx]['labels'].split(';')  # splitting the labels separated for each image;

            for lab in labels:
                for subcat in opts.dense_labels:
                    if subcat in lab:
                        self.dense_label_mapping[subcat].append(idx)
