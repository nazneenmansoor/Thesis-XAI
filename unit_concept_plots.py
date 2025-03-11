# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# unit_concept_plots.py
# Description       : Module which identifies the interpretable concepts for all the facial regions and plots the histogram for all local concepts detected by each model
# Author            : Nazneen Mansoor
# Date              : 25/07/2023
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import math
import numpy as np
import pickle

from settings import get_settings
from matplotlib import pyplot as plt
from tqdm import tqdm
from visual_dict_dataloader import VisDict
from feature_operation import hook_feature, FeatureOperator

def prob_dist():
    data = VisDict('DeepFake_Face_dictionary/', opts.batch_size)
    dm = data.dense_label_mapping

    cluster_image_list = dict((x, []) for x in clusters)

    for region in clusters:
        concepts = clusters[region]
        for cpt in concepts:
            image_list = [x for x in dm[cpt] if not x in cluster_image_list[region]]
            for img_idx in image_list:
                cluster_image_list[region].append(img_idx)

    tally_file = np.load(opts.output_folder + f'/{opts.feature_names[0]}_tally20.npz')
    tally = tally_file['tally']

    cluster_info = {}
    unit_prob_tally = {}

    model_hook = loadmodel(hook_feature)
    feature_constructor = FeatureOperator(opts)
    features, size = feature_constructor.feature_extraction(model=model_hook, face_crop=opts.crop)

    for layer_id, layer in enumerate(opts.feature_names):
        feat = features[layer_id]

    num_units = feat.shape[1]
    unit_ious = {}
    unit_subcats = {}

    for i in range(num_units):
        unit_scores = tally[i]
        subcat_idx = np.argsort(unit_scores)[-1]
        subcat = dense_labels[subcat_idx]

        if unit_scores[subcat_idx] < opts.seg_threshold:
            continue

        unit_ious[i], unit_subcats[i] = unit_scores[subcat_idx], subcat

        for clus in clusters:
            if subcat in clusters[clus]:
                cluster_info[i] = clus

    for unit in tqdm(cluster_info):

        curr_cluster = cluster_info[unit]
        cluster_concepts = clusters[curr_cluster]
        concept_scores = dict((x, 0) for x in cluster_concepts)
        unit_maps = feat[:, unit, :, :]
        range_min = unit_maps.min()
        range_max = unit_maps.max()
        h, w = unit_maps.shape[1], unit_maps.shape[2]
        num_cluster_images = len(cluster_image_list[curr_cluster])
        selected_maps = np.zeros((num_cluster_images, h, w))

        curr_image_list = cluster_image_list[curr_cluster]
        image_mapping = {}

        for i, img_idx in enumerate(curr_image_list):
            selected_maps[i] = unit_maps[img_idx]
            image_mapping[i] = img_idx

        sorted_index = np.argsort(np.max(selected_maps.reshape(num_cluster_images, -1), axis=1))

        for i, idx in enumerate(sorted_index):
            rank = (i + 1) / num_cluster_images
            fmap_score = (selected_maps[idx].max() - range_min) / (range_max - range_min)
            for cpt in cluster_concepts:
                if image_mapping[idx] in dm[cpt]:
                    concept_scores[cpt] += (rank * fmap_score)

        concept_dict = concept_scores.copy()
        for cpt in cluster_concepts:
            try:
                concept_dict[cpt] = concept_scores[cpt] / len(dm[cpt])
            except ZeroDivisionError:
                concept_dict[cpt] = 0

        concept_attributes = concept_dict.copy()
        for cpt in concept_dict:

            concept_value = concept_dict[cpt]
            if not math.isnan(concept_value):
                concept_attributes[cpt] = concept_value
            else:
                concept_attributes[cpt] = 0

        list_values = list(concept_attributes.values())
        sum_values = sum(list(concept_attributes.values()))

        probs = [i / sum_values if sum_values else 0 for i in list_values]

        keys = list(concept_attributes.keys())
        unit_prob_tally[unit] = dict((x, y) for x, y in zip(keys, probs))

    with open('{}/{}_unit_concept_probs1.pkl'.format(opts.output_folder, opts.feature_names[0]), 'wb') as handle:
        pickle.dump(unit_prob_tally, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for j in range(num_units):

        if j in unit_prob_tally.keys():
            outp = ('Unit: {}, Concept: {}, IoU: {}, Cluster: {} {} '.format(j + 1, unit_subcats[j], unit_ious[j],
                                                                             cluster_info[j], unit_prob_tally[j]))
            with open('{}/prob_results/DD1_InceptionV3_prob1_100_images.txt'.format(opts.output_folder), 'a') as file:
                file.write(outp)
                file.write('\n')


def plot_hist():
    tally_region = dict((x, 0) for x in clusters)

    num_units = np.load(opts.output_folder + '/' + '{}_tally20.npz'.format(opts.feature_names[0]))['tally'].shape[0]

    tally_og = dict((x.replace('_', ' '), 0) for x in dense_labels)

    with open('{}/{}_unit_concept_probs1.pkl'.format(opts.output_folder, opts.feature_names[0]), 'rb') as handle:
        data_og = pickle.load(handle)
        tally_region['Unlocalizable'] = num_units - len(data_og)

    for unit in list(data_og.keys()):

        probs_og = data_og[unit]

        for og_concept in probs_og:

            if probs_og[og_concept] > 0:

                tally_og[og_concept.replace('_', ' ')] += 1
                for clus in clusters:
                    if og_concept in clusters[clus]:
                        tally_region[clus] += 1
                        break

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)

    ax.bar(dense_labels, list(tally_og.values()), color='cornflowerblue', edgecolor='black', width=0.4)
    ax.tick_params(rotation=90, labelsize=10)
    ax.set_ylabel('Number of Interpretable Units')
    ax.set_title('Network Dissection - InceptionV3')

    plt.tight_layout()

    fig.savefig('{}/plots/{}_hist.png'.format(opts.output_folder, opts.feature_names[0]))
    plt.close()

    fig = plt.figure(figsize=(12, 5))
    region = list(tally_region.keys())
    region_units = list(tally_region.values())
    print(region_units)

    # creating the bar plot
    plt.bar(region, region_units, color='grey', width=0.2)
    plt.xlabel("Facial regions")
    plt.ylabel("Number of interpretable units")
    plt.savefig('{}/plots/compare/Region_comparison.png'.format(opts.output_folder))
    plt.close()


if __name__ == '__main__':
    opts = get_settings()

    # loading model based on different architectures
    if opts.model == 'VGG-16':
        from model_loader_vgg16_deepfake import loadmodel
    elif opts.model == 'ResNet-50':
        from model_loader_resnet50_deepfake import loadmodel
    elif opts.model == 'Inception-V3':
        from model_loader_inceptionv3_deepfake import loadmodel
    else:
        print('Invalid Model Choice')

    # facial regions based on different facial concepts within same region
    clusters = {'Eye_Region': ['l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g'],
                'Cheek_Region': ['skin'],
                'Nose_Region': ['nose'],
                'Mouth_Region': ['u_lip', 'mouth', 'l_lip'],
                'Neck_Region': ['neck'],
                'Cloth_Region': ['cloth'],
                'Ear_Region': ['l_ear', 'r_ear', 'ear_r'],
                'Hair_Region': ['hair', 'hat']}

    dense_labels = ['hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye', 'skin', 'u_lip', 'cloth', 'l_ear', 'r_ear', 'hat', 'ear_r', 'eye_g']

    prob_dist()
    plot_hist()
