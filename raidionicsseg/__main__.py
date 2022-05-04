import getopt
import os
import sys
import traceback
import argparse
from raidionicsseg.Utils.configuration_parser import ConfigResources
from raidionicsseg.fit import run_model #segment, classify
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='config', type=path, help='Path to the configuration file (*.ini)')

    argsin = sys.argv[1:]
    args = parser.parse_args(argsin)

    config_filename = args.config
    # ConfigResources.getInstance().init_environment(config_filename)

    try:
        run_model(config_filename=config_filename)
        # segment()
    #     if task == 'segmentation':
    #         predict(input_filename=input_filename, output_path=output_prefix, selected_model=model_name)
    #     elif task == 'classification':
    #         classify(input_filename=input_filename, output_path=output_prefix, selected_model=model_name)
    #     else:
    #         raise AttributeError('Wrong task provided. Only [parsing, segmentation] are eligible.')
    except Exception as e:
        print('{}'.format(traceback.format_exc()))


if __name__ == "__main__":
    print("Called as script")
    main()

