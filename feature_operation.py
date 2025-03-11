# -----------------------------------------------------------------------------------------------------------------------------------------------------
# feature_operation.py
# Description       : Module which generated two files (Conv2d.mmap and feature_size.npy) and stored in 'result' folder given in settings.OUTPUT_FOLDER
# Author            : Nazneen Mansoor
# Date              : 01/07/2023
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import concurrent.futures
import os
import time
from typing import List, Tuple, Dict

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

import vecquantile as vecquantile
from visual_dict_dataloader import VisDict

# global variable for features from deep detector model
FEATURES_BLOBS = []


def hook_feature(
        module: torch.nn.Module,
        input: Tuple[torch.Tensor],
        output: Tuple[torch.Tensor],
) -> List[float]:
    """Extract hooked features from model for a specific layer and store in list.

    Args:
        output: hooked features from deep detector model
    Return:
        FEATURES_BLOBS: list of features
    """

    FEATURES_BLOBS.append(output.data.cpu().numpy())


class FeatureOperator():
    """
    Generates two files memory map and features file.
    After generating the features, threshold values are computed and stored in quantile file.
    Then, IoU scores are calculated using tally function and generates the network dissection report.
    """

    def __init__(
            self,
            opts,
    ):
        self.opts = opts
        if not os.path.exists(self.opts.output_folder):
            os.makedirs(os.path.join(opts.output_folder, 'image'))

        self.loader = VisDict('DeepFake_Face_dictionary/', self.opts.batch_size)
        self.tally_loader = VisDict('DeepFake_Face_dictionary/', self.opts.tally_batch_size)

    def feature_extraction(
            self,
            model=None,
            memmap: bool = True,
            face_crop: bool = False,
    ) -> Tuple[List[float], List[float]]:

        """Extract whole features from deep detector model.

        Args:
            hook: hooked feature from deep detector model
            memmap: set memory mapping to True
            face_crop: set face crop to False
        Return:
            whole_features: whole features from model
            size_features: size of features
        """

        # extract the max value activation for each image
        whole_features = [None] * len(self.opts.feature_names)
        features_size = [None] * len(self.opts.feature_names)
        # features_size_file = os.path.join(opts.output_folder, 'feature_size.npy')
        features_size_file = os.path.join(self.opts.output_folder + f'/{self.opts.feature_names[0]}_feature_size.npy')

        # creating memory map file if memmap is True
        if memmap:
            skip = True
            mmap_files = [os.path.join(self.opts.output_folder, '%s.mmap' % f_name)
                          for f_name in self.opts.feature_names]

            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file, allow_pickle=True)
            else:
                skip = False

            # loop through map file for storing features
            for i, mmap_file in enumerate(mmap_files):
                # check if mmap exists and load features into whole_features
                if os.path.exists(mmap_file) and features_size[i] is not None:
                    print(f'loading features {self.opts.feature_names[i], features_size[i]}')
                    whole_features[i] = np.memmap(mmap_file, dtype=float, mode='r', shape=tuple(features_size[i]))
                else:
                    print('file missing, loading from scratch')  # displays if mmap file is missing
                    skip = False

            # if skip=True, feature_extraction function returns whole features and size of features
            if skip:
                return whole_features, features_size

        # invoking get_batches from visual_dict_dataloader to get number of batches
        gen = self.loader.get_batches(only_image=True, face_crop=face_crop)
        num_batches = self.loader.num_batches

        # loop through number of batches to extract feature from each batch
        for batch_idx in tqdm(range(num_batches), desc='Extracting features'):
            del FEATURES_BLOBS[:]
            batch = next(gen)
            model_inp = torch.cat([x.unsqueeze(0) for x in batch])

            logit = model.forward(model_inp)

            if len(FEATURES_BLOBS[0].shape) == 4 and whole_features[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(FEATURES_BLOBS):
                    size_features = (
                    len(self.loader.images), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        whole_features[i] = np.memmap(mmap_files[i], dtype=float, mode='w+', shape=size_features)
                    else:
                        whole_features[i] = np.zeros(size_features)

                np.save(features_size_file, features_size)

            start_idx = batch_idx * self.opts.batch_size
            end_idx = min((batch_idx + 1) * self.opts.batch_size, len(self.loader.images))

            for i, feat_batch in enumerate(FEATURES_BLOBS):
                whole_features[i][start_idx:end_idx] = feat_batch

        return whole_features, size_features

    def quantile_threshold(
            self,
            features: List[float],
            save_path: str = '',
    ) -> List[float]:

        """calculate quantile threshold

        Args:
            features: features extracted from model
            save_path: path name of quantile
        Return:
            ret: quantile threshold values
        """

        quant_path = os.path.join(self.opts.output_folder, save_path)  # adds quantile path into output folder
        if save_path and os.path.exists(quant_path):
            return np.load(quant_path)
        print('calculating quantile threshold')
        # calling QuantileVector class from vecquantile
        quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
        start_time = time.time()
        last_batch_time = start_time
        batch_size = 64

        for i in range(0, features.shape[0], batch_size):
            batch_time = time.time()
            rate = i / (batch_time - start_time + 1e-15)
            batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time
            batch = features[i:i + batch_size]
            batch = np.transpose(batch, axes=(0, 2, 3, 1)).reshape(-1, features.shape[1])
            quant.add(batch)

        # quantile threshold
        ret = quant.readout(1000)[:, int(1000 * (1 - self.opts.quantile) - 1)]
        if save_path:
            np.save(quant_path, ret)
        return ret

    def tally(
            self,
            features: List[float],
            fsize: List[float],
            thresholds: List[float],
            use_crop_points: bool = False,
    ) -> None:

        """calculate IoU scores and generate report

        Args:
            features: features extracted from model
            fsize: size of features
            thresholds: calculated threshold
            use_crop_points: boolean variable for face crop points
        Return: None
        """

        num_units = features.shape[1]
        num_images = features.shape[0]

        labels_map = dict((x, i) for i, x in enumerate(self.opts.dense_labels))  # mapping labels with face labels

        def sample_tally(
                sample: Dict,
        ) -> None:
            """calculate IoU scores for sample images

            Args:
                sample: dictionary with image id and labels
            Return: None
            """
            for unit_idx in range(num_units):
                threshold = thresholds[unit_idx]
                img_idx = sample['i']
                label_set = sample['labels']
                fmap = features[img_idx, unit_idx, :, :]
                fmap = cv2.resize(fmap, (self.opts.input_size, self.opts.input_size))

                if fmap.max() > threshold:
                    indexes = (fmap > threshold).astype(int)
                    thresh_fmap = fmap * indexes
                    for lab in label_set:
                        label_map = label_set[lab]
                        label_map = np.squeeze(label_map.numpy().transpose(1, 2, 0), axis=-1)
                        sub_cat = lab
                        intersection = np.logical_and(label_map, thresh_fmap)  # compute intersection for labels
                        union = np.logical_or(label_map, thresh_fmap)  # compute union for labels
                        iou = (np.sum(intersection) / np.sum(union))  # compute IntersectionOverUnion score
                        tally_units_int[unit_idx, labels_map[sub_cat]] += np.sum(intersection)
                        tally_units_uni[unit_idx, labels_map[sub_cat]] += np.sum(union)
                        all_maps[unit_idx][sub_cat].append(img_idx)
                        all_scores[unit_idx][sub_cat].append(iou)
                        top20 = list(units_top20_score[unit_idx][labels_map[sub_cat]])
                        if iou > min(top20) and iou != 0:
                            units_top20_score[unit_idx][labels_map[sub_cat]][top20.index(min(top20))] = iou
                            units_top20_fmaps[unit_idx][labels_map[sub_cat]][top20.index(min(top20))] = img_idx

        if os.path.exists('{}/{}_tally20.npz'.format(self.opts.output_folder, self.opts.feature_names[0])):
            numpy_file = np.load(self.opts.output_folder + f'/{self.opts.feature_names[0]}_tally20.npz')
            final_tally = numpy_file['tally']
            units_top20_fmaps = numpy_file['maps']
            units_top20_score = numpy_file['scores']
        else:
            tally_units_int = np.zeros((num_units, len(self.opts.dense_labels)), dtype=np.float64)
            tally_units_uni = np.zeros((num_units, len(self.opts.dense_labels)), dtype=np.float64)
            all_maps = dict((x, dict((y, []) for y in self.opts.dense_labels)) for x in range(num_units))
            all_scores = dict((x, dict((y, []) for y in self.opts.dense_labels)) for x in range(num_units))
            units_top20_score = np.ones((num_units, len(self.opts.dense_labels), 20)) * -1
            units_top20_fmaps = np.ones((num_units, len(self.opts.dense_labels), 20)) * -1

            if use_crop_points:
                sample_gen = self.tally_loader.get_batches(face_crop=True)
            else:
                sample_gen = self.tally_loader.get_batches()
            for idx in tqdm(range(0, num_images, self.opts.tally_batch_size)):
                sample_batch = next(sample_gen)
                with concurrent.futures.ThreadPoolExecutor() as exec:
                    exec.map(sample_tally, sample_batch)

            tally_units_uni[tally_units_uni == 0] = 1
            final_tally = tally_units_int / tally_units_uni
            np.savez(
                self.opts.output_folder + '/{}_tally20.npz'.format(self.opts.feature_names[0]),
                tally=final_tally,
                maps=units_top20_fmaps,
                scores=units_top20_score
            )

        sort_dict = self.get_sorted(features, final_tally, units_top20_score, units_top20_fmaps, thresholds, num_units)

        sort_indexes = sort_dict['sorted_indexes']
        sort_thresholds = sort_dict['sorted_thresholds']
        sort_final_tally = sort_dict['sorted_final_tally']
        sort_units_top20_fmaps = sort_dict['sorted_units_top20_fmaps']
        sort_units_top20_score = sort_dict['sorted_units_top20_score']
        sort_features = sort_dict['sorted_features']

        for ii, k in enumerate(sort_indexes):
            sort_features[:, ii, :, :] = features[:, k, :, :]
            sort_thresholds[ii] = thresholds[k]
            sort_final_tally[ii] = final_tally[k]
            sort_units_top20_fmaps[ii] = units_top20_fmaps[k]
            sort_units_top20_score[ii] = units_top20_score[k]

        if use_crop_points:
            face_mod = MTCNN(image_size=256)

        self.generate_report(
            sort_dict,
            face_mod,
            use_crop_points,
            num_units,
        )

    def get_sorted(
            self,
            features: List[int],
            final_tally: List[float],
            units_top20_score: List[float],
            units_top20_fmaps: List[float],
            thresholds: List[float],
            num_units: int,
    ) -> Tuple[List[float]]:
        """
        Sorting the features based on IoU scores
        Args:
            features: features
            final_tally: final tally output
            units_top20_score: units score
            units_top20_fmaps: feature map scores/units
            thresholds: threshold value
            num_units: number of units

        Return:
            sorted_dict: dictionary with sorted values of features, final tally, units, and thresholds
        """

        # sorting features based on IoU scores
        sorted_dict = dict()

        sorting_ious = []
        sorted_dict['sorted_features'] = np.zeros(features.shape)
        sorted_dict['sorted_final_tally'] = np.zeros(final_tally.shape)
        sorted_dict['sorted_units_top20_score'] = np.zeros(units_top20_score.shape)
        sorted_dict['sorted_units_top20_fmaps'] = np.zeros(units_top20_fmaps.shape)
        sorted_dict['sorted_thresholds'] = np.zeros(thresholds.shape)

        for ui in range(num_units):
            us = final_tally[ui]
            sci = np.nanargmax(us)
            cc = self.opts.dense_labels[sci]
            iou = us[sci]
            sorting_ious.append(iou)

        sorted_dict['sorted_indexes'] = np.argsort(sorting_ious)[::-1]

        return sorted_dict

    def generate_report(
            self,
            sort_dict: Dict[List[str], List[float]],
            face_mod,
            use_crop_points: bool,
            num_units: int,
    ) -> None:
        """
        Generating network dissection report
        Args:
            sort_dict: dictionary with sorted values
            face_mod: face mode
            use_crop_points: set crop_points
            num_units: number of units

        Return: None
        """
        for d in tqdm(range(2), desc='Drawing Top IoU Features per Unit'):
            sort_indexes = sort_dict['sorted_indexes']
            sort_thresholds = sort_dict['sorted_thresholds']
            sort_final_tally = sort_dict['sorted_final_tally']
            sort_units_top20_fmaps = sort_dict['sorted_units_top20_fmaps']
            sort_units_top20_score = sort_dict['sorted_units_top20_score']
            sort_features = sort_dict['sorted_features']

            # generating network dissection report as pdf
            with PdfPages(self.opts.output_folder + '/{}.pdf'.format(self.opts.feature_names[0])) as pdf:
                num_top_images = self.opts.topn
                print('num of units', num_units)
                for ux in range(0, num_units, 16):
                    fig, ax = plt.subplots(4, 1)

                    if ux == 0:
                        fig.suptitle('Face NetDissect Results')  # title for network dissection report
                    save_pdf = False
                    for k in range(0, 4):
                        threshold = sort_thresholds[ux + k]
                        unit_scores = sort_final_tally[ux + k]
                        subcat_idx = np.nanargmax(unit_scores)
                        curr_class = self.opts.dense_labels[subcat_idx]
                        overall_iou = unit_scores[subcat_idx]  # overall IoU score

                        # if overall_iou > 0.4:
                        #     overall_iou_final = overall_iou

                        units_top_idx = np.argsort(sort_units_top20_score[ux + k][subcat_idx])[-num_top_images:]
                        fmap_idxs = [sort_units_top20_fmaps[ux + k][subcat_idx][x] for x in units_top_idx]
                        fmap_idxs = [int(x) for x in fmap_idxs if int(x) != -1]

                        tile = np.zeros((self.opts.input_size, self.opts.input_size * num_top_images, 3),
                                        dtype=np.uint8)

                        if len(fmap_idxs) > 0:

                            save_pdf = True
                            fmaps = [cv2.resize(sort_features[x][ux + k], (self.opts.input_size, self.opts.input_size))
                                     for x in
                                     fmap_idxs]

                            crop_images, images, new_points = self.image_preprocess(
                                fmap_idxs,
                                use_crop_points,
                                face_mod,
                            )
                            for i in range(len(fmaps)):
                                if use_crop_points:
                                    img = crop_images[i]
                                else:
                                    img = images[i]

                                img_copy = img.copy()
                                output = img.copy()
                                indexes = (fmaps[i] > threshold).astype(int)
                                thresh_map = fmaps[i] * indexes
                                img_copy[thresh_map <= 0] = 0
                                cv2.addWeighted(img, 0.4, img_copy, 0.6, 0, output)

                                if use_crop_points:
                                    full_img = images[i].astype(np.uint8)
                                    overlay = np.zeros((full_img.shape)).astype(np.uint8)
                                    full_output = cv2.addWeighted(full_img, 0.4, overlay, 0.6, 0)
                                    x1, y1, x2, y2 = new_points[i]
                                    output = cv2.resize(output, (x2 - x1, y2 - y1))
                                    full_output[y1:y2, x1:x2] = output
                                    output = full_output

                                tile[:, self.opts.input_size * 1 * i: self.opts.input_size * 1 * (i + 1), :] = output

                            ax[k].imshow(tile[:, :, ::-1])
                            ax[k].set_title(
                                'UNIT - {},  CLASS - {}, IOU - {:.4f}'.format(sort_indexes[ux + k],
                                                                              curr_class,
                                                                              overall_iou),
                                fontsize=7)
                            ax[k].axes.xaxis.set_visible(False)
                            ax[k].axes.yaxis.set_visible(False)

                    if save_pdf:
                        pdf.savefig()
                        plt.close()
                    else:
                        plt.close()

    def image_preprocess(
            self,
            fmap_idxs: List[int],
            use_crop_points: bool,
            face_mod,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Reading and preprocessing images from input folder
        Args:
            fmap_idxs: feature map ids
            use_crop_points: boolean value to specify cropping
            face_mod: face mode

        Returns:
            cropped_images: images after resizing and cropping
            images_input: input images
            new_points1: facial points

        """
        # reading images from input folder
        images_input = [cv2.resize(cv2.imread('DeepFake_Face_dictionary/' + self.loader.images[x]['image']),
                                   (self.opts.input_size, self.opts.input_size)) for x in fmap_idxs]
        # converting the images into RGB format
        pil_images = [
            Image.open('DeepFake_Face_dictionary/' + self.loader.images[x]['image']).convert('RGB')
                .resize((self.opts.input_size, self.opts.input_size)) for x in fmap_idxs]
        cropped_images = [None] * len(images_input)

        if use_crop_points:
            points = [face_mod.detect(x.copy(), landmarks=False) for x in pil_images]
            points = [x[0] for x in points]
            new_points1 = [None] * len(points)
            NoneType = type(None)
            for ix, pt in enumerate(points):
                if isinstance(pt, NoneType):
                    new_pt = np.array([0, 0, self.opts.input_size, self.opts.input_size])
                    new_points1[ix] = new_pt
                    curr_img = cv2.resize(images_input[ix], (self.opts.input_size, self.opts.input_size))
                    cropped_images[ix] = curr_img
                else:
                    new_pt = pt[0]
                    new_pt = [int(x) for x in new_pt]
                    new_pt[0] = max(0, new_pt[0])
                    new_pt[1] = max(0, new_pt[1])
                    new_pt[2] = min(self.opts.input_size, new_pt[2])
                    new_pt[3] = min(self.opts.input_size, new_pt[3])
                    new_points1[ix] = new_pt
                    cropped_images[ix] = cv2.resize(
                        images_input[ix][new_pt[1]:new_pt[3], new_pt[0]:new_pt[2]],
                        (self.opts.input_size, self.opts.input_size))

        return cropped_images, images_input, new_points1
