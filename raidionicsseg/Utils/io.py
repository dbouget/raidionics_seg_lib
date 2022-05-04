import os
import nibabel as nib
from nibabel import four_to_three
import SimpleITK as sitk
import numpy as np
# import pandas as pd
from tensorflow.python.keras.models import Model


def load_nifti_volume(volume_path):
    nib_volume = nib.load(volume_path)
    if len(nib_volume.shape) > 3:
        if len(nib_volume.shape) == 4: #Common problem
            nib_volume = four_to_three(nib_volume)[0]
        else: #DWI volumes
            nib_volume = nib.Nifti1Image(nib_volume.get_data()[:, :, :, 0, 0], affine=nib_volume.affine)

    return nib_volume


def dump_predictions(predictions, parameters, nib_volume, storage_path):
    print("Writing predictions to files...")
    naming_suffix = 'pred' if parameters.predictions_reconstruction_method == 'probabilities' else 'labels'
    class_names = parameters.training_class_names

    if len(predictions.shape) == 4:
        for c in range(1, predictions.shape[-1]):
            img = nib.Nifti1Image(predictions[..., c], affine=nib_volume.affine)
            predictions_output_path = os.path.join(storage_path, naming_suffix + '_' + class_names[c] + '.nii.gz')
            os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
            nib.save(img, predictions_output_path)
    else:
        img = nib.Nifti1Image(predictions, affine=nib_volume.affine)
        predictions_output_path = os.path.join(storage_path, naming_suffix + '_' + 'argmax' + '.nii.gz')
        os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
        nib.save(img, predictions_output_path)


def dump_classification_predictions(predictions, parameters, storage_path):
    print("Writing predictions to files...")
    class_names = parameters.training_class_names
    prediction_filename = os.path.join(storage_path, 'classification-results.csv')

    # @TODO. Including pandas just for this might not be necessary?
    # results_df = pd.DataFrame(np.expand_dims(predictions, axis=0), columns=class_names)
    # results_df.to_csv(prediction_filename, index=False)


def convert_and_export_to_nifti(input_filepath):
    input_sitk = sitk.ReadImage(input_filepath)
    output_filepath = input_filepath.split('.')[0] + '.nii.gz'
    sitk.WriteImage(input_sitk, output_filepath)

    return output_filepath


#-t segmentation -i /media/dbouget/ihdb/Data/NeuroDatabase/20/3860/volumes/3860_MR_T1_pre_4778.nii.gz -o ./Results -m MRI_HGGlioma -g -1
#-t segmentation -i /media/dbouget/ihdb/Data/NeuroDatabase/4/722/volumes/722_MR_T1_pre_887.nii.gz -o ./Results -m MRI_Meningioma -g -1
#-t segmentation -i /media/dbouget/ihdb/Data/NeuroDatabase/7/1257/volumes/1257_MR_T1_pre_1518.nii.gz -o ./Results -m MRI_Meningioma -g -1
def dump_feature_maps(model, data_prep):
    output_root = '/media/dbouget/ihda/Studies/NetworkValidation/Meningioma/Viz'
    # output_root = '/media/dbouget/ihda/Studies/NetworkValidation/G2Paper/Viz'
    affine_default = [[1,0,0,0], [0,1,0,0], [0,0,1,0],[0,0,0,0]]
    nib.save(nib.Nifti1Image(data_prep[0, :, :, :, 0], affine=affine_default), os.path.join(output_root, 'prep_input.nii.gz'))
    layer_names = [layer.name for layer in model.layers]
    layer_outputs = [layer.output for layer in model.layers]
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if 'conv' not in layer.name:
            continue
        print(i, layer.name, layer.output.shape)

    blocks = [83, 103, 108, 111, 114, 117, 118, 123, 128, 131, 134] # [63, 66, 71, 74, 77]
    outputs = [model.layers[i].output for i in blocks]

    model2 = Model(inputs=model.inputs, outputs=outputs)
    feature_map = model2.predict(data_prep)
    for i, fmap in zip(blocks, feature_map):
        # Number of feature images/dimensions in a feature map of a layer
        k = fmap.shape[-1]
        for fm in range(k):
            output_filename = os.path.join(output_root, 'Layer_' + str(i) + '_fm' + str(fm) + '.nii.gz')
            if len(fmap.shape) == 4:
                curr_fm = fmap[:, :, :, fm]
            elif len(fmap.shape) == 5:
                curr_fm = fmap[0, :, :, :, fm]
            # mean = curr_fm.mean()
            # std = curr_fm.std()
            # curr_fm -= mean
            # curr_fm /= std
            min = curr_fm.min()
            max = curr_fm.max()
            curr_fm -= min
            curr_fm /= (max - min)
            layer_viz_ni = nib.Nifti1Image(curr_fm, affine=affine_default)
            nib.save(layer_viz_ni, output_filename)

    # for layer_name, layer_output in zip(layer_names, layer_outputs):
    #     # Number of feature images/dimensions in a feature map of a layer
    #     k = layer_output.shape[-1]
    #     for fm in range(k):
    #         output_filename = os.path.join(output_root, layer_name + '_fm' + str(fm) + '.nii.gz')
    #         feature_map = layer_output[0, :, :, :, fm]
    #         feature_map -= feature_map.mean()
    #         feature_map /= feature_map.std()
    #         feature_map *= 64
    #         feature_map += 128
    #         feature_map = np.clip(feature_map, 0, 255).astype('uint8')
    #         layer_viz_ni = nib.Nifti1Image(feature_map, affine=affine_default)
    #         nib.save(layer_viz_ni, output_filename)
