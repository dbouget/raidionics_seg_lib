import configparser
import os
import sys
import logging
from aenum import Enum, unique


@unique
class ImagingModalityType(Enum):
    _init_ = 'value string'

    CT = 0, 'CT'
    MRI = 1, 'MRI'
    US = 2, 'US'

    def __str__(self):
        return self.string


def get_type_from_string(EnumType, string):
    if type(string) == str:
        for i in range(len(list(EnumType))):
            if string == str(list(EnumType)[i]):
                return list(EnumType)[i]
        return -1
    elif type(string) == EnumType:
        return string
    else:  # Un-managed input type
        return -1


class ConfigResources:
    """
    Class defining and holding the various (user-specified) configuration and runtime parameters.
    """
    def __init__(self):
        self.__setup()

    def __setup(self):
        self.config_filename = None
        self.config = None

        self.gpu_id = "-1"
        self.inputs_folder = None
        self.output_folder = None
        self.model_folder = None

        self.predictions_non_overlapping = True
        self.predictions_folds_ensembling = False
        self.predictions_ensembling_strategy = "average"
        self.predictions_overlapping_ratio = None
        self.predictions_reconstruction_method = None
        self.predictions_reconstruction_order = None
        self.predictions_use_preprocessed_data = False
        self.predictions_test_time_augmentation_iterations = 0
        self.predictions_test_time_augmentation_fusion_mode = "average"

        self.runtime_lungs_mask_filepath = ''
        self.runtime_brain_mask_filepath = ''

    def init_environment(self, config_filename):
        self.config_filename = config_filename
        self.config = configparser.ConfigParser()
        self.config.read(self.config_filename)
        self.__parse_main_config()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        self.preprocessing_filename = os.path.join(self.model_folder, 'pre_processing.ini')
        if not os.path.exists(self.preprocessing_filename):
            raise ValueError('Missing configuration file with pre-processing parameters: {}'.
                             format(self.preprocessing_filename))

        self.pre_processing_config = configparser.ConfigParser()
        self.pre_processing_config.read(self.preprocessing_filename)
        self.__parse_content()

    def __parse_main_config(self):
        if self.config.has_option('System', 'gpu_id'):
            if self.config['System']['gpu_id'].split('#')[0].strip() != '':
                self.gpu_id = self.config['System']['gpu_id'].split('#')[0].strip()

        if self.config.has_option('System', 'inputs_folder'):
            if self.config['System']['inputs_folder'].split('#')[0].strip() != '':
                self.inputs_folder = self.config['System']['inputs_folder'].split('#')[0].strip()

        if self.config.has_option('System', 'output_folder'):
            if self.config['System']['output_folder'].split('#')[0].strip() != '':
                self.output_folder = self.config['System']['output_folder'].split('#')[0].strip()

        if self.config.has_option('System', 'model_folder'):
            if self.config['System']['model_folder'].split('#')[0].strip() != '':
                self.model_folder = self.config['System']['model_folder'].split('#')[0].strip()

        if self.config.has_option('Runtime', 'non_overlapping'):
            if self.config['Runtime']['non_overlapping'].split('#')[0].strip() != '':
                self.predictions_non_overlapping = True if self.config['Runtime']['non_overlapping'].split('#')[0].lower().strip()\
                                                       == 'true' else False

        if self.config.has_option('Runtime', 'folds_ensembling'):
            if self.config['Runtime']['folds_ensembling'].split('#')[0].strip() != '':
                self.predictions_folds_ensembling = True if self.config['Runtime']['folds_ensembling'].split('#')[0].lower().strip()\
                                                       == 'true' else False

        if self.config.has_option('Runtime', 'ensembling_strategy'):
            if self.config['Runtime']['ensembling_strategy'].split('#')[0].strip() != '':
                self.predictions_ensembling_strategy = self.config['Runtime']['ensembling_strategy'].split('#')[0].lower().strip()
        if self.predictions_ensembling_strategy not in ["maximum", "average"]:
            self.predictions_ensembling_strategy = "average"
            logging.warning("""Value provided in [Runtime][ensembling_strategy] is not recognized.
             setting to default parameter with value: {}""".format(self.predictions_ensembling_strategy))

        if self.config.has_option('Runtime', 'overlapping_ratio'):
            if self.config['Runtime']['overlapping_ratio'].split('#')[0].strip() != '':
                self.predictions_overlapping_ratio = float(self.config['Runtime']['overlapping_ratio'].split('#')[0].lower().strip())

        if self.config.has_option('Runtime', 'reconstruction_method'):
            if self.config['Runtime']['reconstruction_method'].split('#')[0].strip() != '':
                self.predictions_reconstruction_method = self.config['Runtime']['reconstruction_method'].split('#')[0].strip()

        if self.config.has_option('Runtime', 'reconstruction_order'):
            if self.config['Runtime']['reconstruction_order'].split('#')[0].strip() != '':
                self.predictions_reconstruction_order = self.config['Runtime']['reconstruction_order'].split('#')[0].strip()

        if self.config.has_option('Runtime', 'use_preprocessed_data'):
            if self.config['Runtime']['use_preprocessed_data'].split('#')[0].strip() != '':
                self.predictions_use_preprocessed_data = True if self.config['Runtime']['use_preprocessed_data'].split('#')[0].strip().lower() == 'true' else False

        if self.config.has_option('Runtime', 'test_time_augmentation_iteration'):
            if self.config['Runtime']['test_time_augmentation_iteration'].split('#')[0].strip() != '':
                self.predictions_test_time_augmentation_iterations = int(self.config['Runtime']['test_time_augmentation_iteration'].split('#')[0].strip())

        if self.config.has_option('Runtime', 'test_time_augmentation_fusion_mode'):
            if self.config['Runtime']['test_time_augmentation_fusion_mode'].split('#')[0].strip() != '':
                self.predictions_test_time_augmentation_fusion_mode = self.config['Runtime']['test_time_augmentation_fusion_mode'].split('#')[0].strip().lower()
        if self.predictions_test_time_augmentation_fusion_mode not in ["maximum", "average"]:
            self.predictions_test_time_augmentation_fusion_mode = "average"
            logging.warning("""Value provided in [Runtime][test_time_augmentation_fusion_mode] is not recognized.
             setting to default parameter with value: {}""".format(self.predictions_test_time_augmentation_fusion_mode))

        if self.config.has_option('Neuro', 'brain_segmentation_filename'):
            if self.config['Neuro']['brain_segmentation_filename'].split('#')[0].strip() != '':
                self.runtime_brain_mask_filepath = self.config['Neuro']['brain_segmentation_filename'].split('#')[0].strip()

        if self.config.has_option('Mediastinum', 'lungs_segmentation_filename'):
            if self.config['Mediastinum']['lungs_segmentation_filename'].split('#')[0].strip() != '':
                self.runtime_lungs_mask_filepath = self.config['Mediastinum']['lungs_segmentation_filename'].split('#')[0].strip()

    def __parse_content(self):
        self.__parse_default_content()
        self.__parse_pre_processing_content()
        self.__parse_training_content()
        self.__parse_MRI_content()
        self.__parse_CT_content()

    def __parse_default_content(self):
        self.training_backend = 'TF'
        self.imaging_modality = "MRI"
        self.task = "segmentation"

        if self.pre_processing_config.has_option('Default', 'imaging_modality'):
            param = self.pre_processing_config['Default']['imaging_modality'].split('#')[0].strip()
            modality = get_type_from_string(ImagingModalityType, param)
            if modality == -1:
                raise AttributeError('')
            self.imaging_modality = modality
        else:
            raise AttributeError('')

        if self.pre_processing_config.has_option('Default', 'training_backend'):
            if self.pre_processing_config['Default']['training_backend'].split('#')[0].strip() != '':
                self.training_backend = self.pre_processing_config['Default']['training_backend'].split('#')[0].strip()

        if self.pre_processing_config.has_option('Default', 'task'):
            if self.pre_processing_config['Default']['task'].split('#')[0].strip() != '':
                self.task = self.pre_processing_config['Default']['task'].split('#')[0].strip()

    def __parse_training_content(self) -> None:
        """
        Configuration parameters specifying the training procedure.


        Returns
        -------

        """
        self.training_nb_classes = None
        self.training_class_names = None
        self.training_slab_size = None
        self.training_patch_size = None
        self.training_patch_offset = None
        self.training_optimal_thresholds = None
        self.training_deep_supervision = False
        self.training_activation_layer_included = True
        self.training_activation_layer_type = "softmax"

        if self.pre_processing_config.has_option('Training', 'nb_classes'):
            self.training_nb_classes = int(self.pre_processing_config['Training']['nb_classes'].split('#')[0])

        if self.pre_processing_config.has_option('Training', 'classes'):
            if self.pre_processing_config['Training']['classes'].split('#')[0].strip() != '':
                self.training_class_names = [x.strip() for x in self.pre_processing_config['Training']['classes'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('Training', 'slab_size'):
            if self.pre_processing_config['Training']['slab_size'].split('#')[0].strip() != '':
                self.training_slab_size = int(self.pre_processing_config['Training']['slab_size'].split('#')[0])

        if self.pre_processing_config.has_option('Training', 'patch_size'):
            if self.pre_processing_config['Training']['patch_size'].split('#')[0].strip() != '':
                self.training_patch_size = [int(x) for x in self.pre_processing_config['Training']['patch_size'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('Training', 'patch_offset'):
            if self.pre_processing_config['Training']['patch_offset'].split('#')[0].strip() != '':
                self.training_patch_offset = [int(x) for x in self.pre_processing_config['Training']['patch_offset'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('Training', 'optimal_thresholds'):
            if self.pre_processing_config['Training']['optimal_thresholds'].split('#')[0].strip() != '':
                self.training_optimal_thresholds = [float(x.strip()) for x in self.pre_processing_config['Training']['optimal_thresholds'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('Training', 'deep_supervision'):
            if self.pre_processing_config['Training']['deep_supervision'].split('#')[0].strip() != '':
                self.training_deep_supervision = True if self.pre_processing_config['Training']['deep_supervision'].split('#')[0].strip().lower() == 'true' else False

        if self.pre_processing_config.has_option('Training', 'activation_layer_included'):
            if self.pre_processing_config['Training']['activation_layer_included'].split('#')[0].strip() != '':
                self.training_activation_layer_included = True if self.pre_processing_config['Training']['activation_layer_included'].split('#')[0].strip().lower() == 'true' else False

        if self.pre_processing_config.has_option('Training', 'activation_layer_type'):
            if self.pre_processing_config['Training']['activation_layer_type'].split('#')[0].strip() != '':
                self.training_activation_layer_type = self.pre_processing_config['Training']['activation_layer_type'].split('#')[0].strip()

    def __parse_pre_processing_content(self) -> None:
        """
        Configuration parameters relating to the data preprocessing. The pre_processing.ini file containing the
        parameters is filled by hand and placed alongside the model file.

        preprocessing_library: Python package used for resampling/resizing, to sample from [nibabel]
        output_spacing: Comma-separated list of floats indicating the desired output spacing
        crop_background: Strategy for removing non-useful background. To sample from: [None, minimum, brain_mask,
         brain_clip, lungs_mask, lungs_clip]
        intensity_clipping_values: Intensity range to select by values (CT use-case). E.g., [-1024, 1024]
        intensity_clipping_range: Intensity range to select by percentage (MRI use-case). E.g., [0, 99.5] to nullify the highest 0.5% of intensity values
        intensity_target_range: If a specific values range, after normalization/standardization, should be enforced
        normalization_method: Method to use for intensity normalization, to sample from: [None, default, zeromean]
        preprocessing_number_inputs: Integer indicating the amount of input channels
        preprocessing_channels_order: String indicating if channels_first or channels_last

        Returns
        -------

        """
        self.preprocessing_library = 'nibabel'
        self.output_spacing = None
        self.crop_background = None
        self.intensity_clipping_values = None
        self.intensity_clipping_range = [0.0, 100.0]
        self.intensity_target_range = [0.0, 1.0]
        self.new_axial_size = None
        self.slicing_plane = 'axial'
        self.swap_training_input = False
        self.normalization_method = None
        self.preprocessing_number_inputs = None
        self.preprocessing_inputs_sub_indexes = None
        self.preprocessing_channels_order = 'channels_last'

        if self.pre_processing_config.has_option('PreProcessing', 'output_spacing'):
            if self.pre_processing_config['PreProcessing']['output_spacing'].split('#')[0].strip() != '':
                self.output_spacing = [float(x) for x in self.pre_processing_config['PreProcessing']['output_spacing'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'intensity_clipping_values'):
            if self.pre_processing_config['PreProcessing']['intensity_clipping_values'].split('#')[0].strip() != '':
                self.intensity_clipping_values = [float(x) for x in self.pre_processing_config['PreProcessing']['intensity_clipping_values'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'intensity_clipping_range'):
            if self.pre_processing_config['PreProcessing']['intensity_clipping_range'].split('#')[0].strip() != '':
                self.intensity_clipping_range = [float(x) for x in self.pre_processing_config['PreProcessing']['intensity_clipping_range'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'intensity_final_range'):
            if self.pre_processing_config['PreProcessing']['intensity_final_range'].split('#')[0].strip() != '':
                self.intensity_target_range = [float(x) for x in self.pre_processing_config['PreProcessing']['intensity_final_range'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'background_cropping'):
            if self.pre_processing_config['PreProcessing']['background_cropping'].split('#')[0].strip() != '':
                self.crop_background = self.pre_processing_config['PreProcessing']['background_cropping'].split('#')[0].strip().lower()

        if self.pre_processing_config.has_option('PreProcessing', 'new_axial_size'):
            if self.pre_processing_config['PreProcessing']['new_axial_size'].split('#')[0].strip() != '':
                self.new_axial_size = [int(x) for x in self.pre_processing_config['PreProcessing']['new_axial_size'].split('#')[0].split(',')]

        if self.pre_processing_config.has_option('PreProcessing', 'slicing_plane'):
            if self.pre_processing_config['PreProcessing']['slicing_plane'].split('#')[0].strip() != '':
                self.slicing_plane = self.pre_processing_config['PreProcessing']['slicing_plane'].split('#')[0]

        if self.pre_processing_config.has_option('PreProcessing', 'swap_training_input'):
            self.swap_training_input = True if self.pre_processing_config['PreProcessing']['swap_training_input'].split('#')[0].lower()\
                                                   == 'true' else False
        if self.pre_processing_config.has_option('PreProcessing', 'normalization_method'):
            if self.pre_processing_config['PreProcessing']['normalization_method'].split('#')[0].strip() != '':
                self.normalization_method = self.pre_processing_config['PreProcessing']['normalization_method'].split('#')[0].strip().lower()

        if self.pre_processing_config.has_option('PreProcessing', 'number_inputs'):
            if self.pre_processing_config['PreProcessing']['number_inputs'].split('#')[0].strip() != '':
                self.preprocessing_number_inputs = int(self.pre_processing_config['PreProcessing']['number_inputs'].split('#')[0].strip())

        if self.pre_processing_config.has_option('PreProcessing', 'diffloader_pairs'):
            if self.pre_processing_config['PreProcessing']['diffloader_pairs'].split('#')[0].strip() != '':
                self.preprocessing_inputs_sub_indexes = [[int(x.split(',')[0]), int(x.split(',')[1])] for x in self.pre_processing_config['PreProcessing']['diffloader_pairs'].split('#')[0].strip().split(';')]

        if self.pre_processing_config.has_option('PreProcessing', 'channels_order'):
            if self.pre_processing_config['PreProcessing']['channels_order'].split('#')[0].strip() != '':
                self.preprocessing_channels_order = self.pre_processing_config['PreProcessing']['channels_order'].split('#')[0].strip()

    def __parse_MRI_content(self) -> None:
        """
        Configuration parameters specific to MRI inputs.

        perform_bias_correction: Boolean indicating if the N4 bias correction from ANTs should be used as preprocessing

        Returns
        -------

        """
        self.perform_bias_correction = False

        if self.pre_processing_config.has_option('MRI', 'perform_bias_correction'):
            self.perform_bias_correction = True if self.pre_processing_config['MRI']['perform_bias_correction'].split('#')[0].lower().strip()\
                                                   == 'true' else False

    def __parse_CT_content(self) -> None:
        """
        Configuration inputs specific to CT inputs

        fix_orientation: ??

        Returns
        -------

        """
        self.fix_orientation = False
        if self.pre_processing_config.has_option('CT', 'fix_orientation'):
            self.fix_orientation = True if self.pre_processing_config['CT']['fix_orientation'].split('#')[0].lower().strip()\
                                                   == 'true' else False


