# --------------------------------------------------------------------------------------------------------------------------
# settings.py
# Description       : Module which includes global settings, change flags in this module to determine which model to dissect
# Author            : Nazneen Mansoor
# Date              : 01/07/2023
# ---------------------------------------------------------------------------------------------------------------------------
import argparse

def get_settings():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model selection and threshold options
    parser.add_argument('--model', type=str, default='Inception-V3', help='Model for training deep detector')
    parser.add_argument('--quantile', type=float, default=0.04, help='threshold used for activation')
    parser.add_argument('--seg_threshold', type=float, default=0.04, help='threshold used for visualization')
    parser.add_argument('--score_threshold', type=float, default=0.04, help='threshold used for IoU score')

    # Output folder and layer options
    parser.add_argument('--topn', default=4, help='to show top N images with highest IoU for each unit')
    parser.add_argument('--output_folder', type=str, default='result/Net_dissection_report/InceptionV3/Conv4/ND_report1_th_0.04',
                        help='result will be stored in this folder')
    parser.add_argument('--output_plots', type=str, default='result/Net_dissection_report/Stacked_bar_plots',
                        help='folder to store stacked bar plots')
    parser.add_argument('--feature_names', default=['Conv4'])
    parser.add_argument('--layer', default=4, help='layer selected for network dissection')

    # Basic training options
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--tally_batch_size', type=int, default=32)
    parser.add_argument('--crop', type=str, default='False')

    # labels for spatial concepts
    parser.add_argument('--dense_labels', default='all', help='dense labels for spatial concepts')

    opts = parser.parse_args()

    if opts.dense_labels == 'all':

        opts.dense_labels = ['hair', 'l_brow', 'l_eye', 'l_lip', 'mouth', 'neck', 'nose', 'r_brow', 'r_eye', 'skin', 'u_lip', 'cloth', 'l_ear', 'r_ear', 'hat', 'ear_r', 'eye_g']
        # opts.dense_labels = [
        #     'Pointy_Nose',
        #     # 'left_brow',
        #     # 'right_brow',
        #     # 'left_eye',
        #     # 'right_eye',
        #     'Mouth_Slightly_Open',
        #     'left_cheek',
        #     'right_cheek',
        #     '5_o_Clock_Shadow',
        #     'Arched_Eyebrows',
        #     'Bushy_Eyebrows',
        #     'No_Beard',
        #     'Bags_Under_Eyes',
        #     'Big_Lips',
        #     'Big_Nose',
        #     'Double_Chin',
        #     'Eyeglasses',
        #     'Goatee',
        #     'High_Cheekbones',
        #     'Rosy_Cheeks',
        #     'Mouth_Slightly_Open',
        #     'Mustache',
        #     'Narrow_Eyes',
        #     'Pointy_Nose',
        #     'Smiling',
        #     'Wearing_Lipstick',
        # ]

    return opts
