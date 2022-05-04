import os, sys
from tensorflow.python.keras.models import load_model
import nibabel as nib
from nibabel.processing import *
from nibabel import four_to_three
from raidionicsseg.Utils.volume_utilities import *
from raidionicsseg.Utils.configuration_parser import *
from raidionicsseg.PreProcessing.mediastinum_clipping import *
from math import ceil

MODELS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', 'models')
print(MODELS_PATH)
sys.path.insert(1, MODELS_PATH)


def predict_CT(input_volume_path, output_mask_path, pre_processing_parameters, model_path):
    print("Extracting data...")
    nib_volume = nib.load(input_volume_path)
    if len(nib_volume.shape) > 3:
        nib_volume = four_to_three(nib_volume)[0]

    print("Pre-processing...")
    # Normalize spacing
    new_spacing = pre_processing_parameters.output_spacing
    if pre_processing_parameters.output_spacing[0] == 0:
        tmp = np.min(nib_volume.header.get_zooms())
        new_spacing = [tmp, tmp, tmp]
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    data = resampled_volume.get_data().astype('float32')

    # Exclude background
    crop_bbox = None
    if pre_processing_parameters.crop_background:
        volume, crop_bbox = mediastinum_clipping(volume=data, parameters=pre_processing_parameters)
        data = volume

    # Normalize values
    data = intensity_normalization(volume=data, parameters=pre_processing_parameters)

    # set intensity range
    mins, maxs = pre_processing_parameters.intensity_target_range
    data *= maxs #@TODO: Modify it such that it handles scaling to i.e. [-1, -1]

    data = resize_volume(data, pre_processing_parameters, order=1)

    print("Loading model...")
    model = load_model(model_path, compile=False)

    print("Predicting...")
    # split data into chunks and predict
    slab_size = pre_processing_parameters.slab_size
    final_result = np.zeros_like(data)
    data = np.expand_dims(data, axis=-1)
    count = 0
    temp_multiclass = False

    # whether to do overlapping predictions or not
    if pre_processing_parameters.non_overlapping:
        scale = ceil(data.shape[-2]/slab_size)
        chunks = np.zeros(data.shape[:-2] + (slab_size, scale), dtype=np.float32)
        for chunk in range(scale):
            slab_CT = np.zeros(data.shape[:-2] + (slab_size, ), dtype=np.float32)
            tmp = data[:, :, int(chunk*slab_size):int((chunk + 1)*slab_size), 0]
            slab_CT[:, :, :tmp.shape[2]] = tmp
            slab_CT = np.expand_dims(np.expand_dims(slab_CT, axis=0), axis=-1)

            # set batch-image-ordering
            if pre_processing_parameters.swap_training_input:
                slab_CT = np.transpose(slab_CT, axes=(0, 3, 1, 2, 4))
            slab_CT = model.predict(slab_CT)
            if pre_processing_parameters.swap_training_input:
                slab_CT = np.transpose(slab_CT, axes=(0, 2, 3, 1, 4))
            #@TODO. Have to handle multiclass in a better way.
            if slab_CT.shape[-1] == 2:
                final_result[..., int(chunk*slab_size):int((chunk + 1)*slab_size)] = slab_CT[0, :, :, :tmp.shape[-1], 1]
            else:
                temp_multiclass = True
                final_result[..., int(chunk * slab_size):int((chunk + 1) * slab_size)] = np.argmax(slab_CT[0], axis=3)[:, :, :tmp.shape[-1]]
            print(count)
            count += 1
        # @TODO. Have to consider the multiclass case here too!!
        if pre_processing_parameters.output_threshold and not temp_multiclass:
            final_result = (final_result > pre_processing_parameters.output_threshold).astype(np.uint8)
        del tmp
    else:
        if slab_size == 1:
            for slice in range(0, data.shape[2]):
                slab_CT = data[:, :, slice, 0]
                if np.sum(slab_CT > 0.1) == 0:
                    continue
                slab_CT = model.predict(np.reshape(slab_CT, (1, pre_processing_parameters.new_axial_size[0],
                                                            pre_processing_parameters.new_axial_size[1], 1)))
                slab_CT = np.argmax(slab_CT[0], axis=2)
                final_result[:, :, slice] = slab_CT
                print(count)
                count += 1
        else:
            half_slab_size = int(slab_size/2)
            for slice in range(half_slab_size, data.shape[2] - slab_size):
                slab_CT = data[:, :, slice - half_slab_size:slice + half_slab_size, 0]
                if np.sum(slab_CT > 0.1) == 0:
                    continue
                slab_CT = np.reshape(slab_CT, (1, pre_processing_parameters.new_axial_size[0],
                                                            pre_processing_parameters.new_axial_size[1], slab_size, 1))
                # set batch-image-ordering
                if pre_processing_parameters.swap_training_input:
                    slab_CT = np.transpose(slab_CT, axes=(0, 3, 1, 2, 4))
                slab_CT = model.predict(slab_CT)
                if pre_processing_parameters.swap_training_input:
                    slab_CT = np.transpose(slab_CT, axes=(0, 2, 3, 1, 4))
                slab_CT = np.argmax(slab_CT[0], axis=3)
                #@TODO. Check the line below -- how to handle binary and multiclass segmentation
                #final_result[:, :, slice] = slab_CT[0, :, :, half_slab_size]
                final_result[:, :, slice] = slab_CT[:, :, half_slab_size]
                print(count)
                count += 1
    
    del slab_CT


    print('Begin predictions reconstruction in input space.')
    #data = (final_result >= 0.75).astype(np.uint8) # convert to binary volume
    data = final_result.astype(np.uint8)

    # Undo resizing (which is performed in function crop())
    if crop_bbox is not None:
        data = resize(data, (crop_bbox[3]-crop_bbox[0], crop_bbox[4]-crop_bbox[1], crop_bbox[5]-crop_bbox[2]),
                      order=0, preserve_range=True)
        # Undo cropping (which is performed in function crop())
        new_data = np.zeros(resampled_volume.get_data().shape, dtype=np.float32)
        new_data[crop_bbox[0]:crop_bbox[3], crop_bbox[1]:crop_bbox[4], crop_bbox[2]:crop_bbox[5]] = data
    else:
        new_data = resize(data.astype(np.float32), resampled_volume.get_data().shape, order=0, preserve_range=True) # <- also need to resize without cropping (!)
    del data

    # Save segmentation data adjusted to original size
    print("Writing to file...")
    img = nib.Nifti1Image(new_data.astype(np.uint8), affine=resampled_volume.affine)
    resampled_lab = resample_from_to(img, nib_volume, order=0)


    if nib_volume.shape != resampled_lab.shape:
        print('Aborting -- dimensions are mismatching between input and obtained labels.')

    print('Type of the saved labels: {}'.format(img.get_data_dtype()))
    nib.save(resampled_lab, output_mask_path)
    print(input_volume_path)
    print(output_mask_path)

    print("Finished!")
