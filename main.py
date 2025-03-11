# ------------------------------------------------------------------------------------------------------
# main.py
# Description       : Module which calls feature extraction, model loader, threshold, and tally functions
# Author            : Nazneen Mansoor
# Date              : 01/07/2023
# -------------------------------------------------------------------------------------------------------

from settings import get_settings
from feature_operation import hook_feature, FeatureOperator

if __name__ == '__main__':

    opts = get_settings()
    if opts.model == 'VGG-16':
        from model_loader_vgg16_deepfake import loadmodel
    elif opts.model == 'ResNet-50':
        from model_loader_resnet50_deepfake import loadmodel
    elif opts.model == 'Inception-V3':
        from model_loader_inceptionv3_deepfake import loadmodel
    else:
        print('Invalid Model Choice')

    model_hook = loadmodel(hook_feature)
    feature_constructor = FeatureOperator(opts)
    features, size = feature_constructor.feature_extraction(model=model_hook, face_crop=opts.crop)

    for layer_id, layer in enumerate(opts.feature_names):
        thresholds = feature_constructor.quantile_threshold(features[layer_id], save_path=f'{opts.feature_names[0]}_quantile.npy')
        tally_result = feature_constructor.tally(features[layer_id], size, thresholds, use_crop_points=opts.crop)
