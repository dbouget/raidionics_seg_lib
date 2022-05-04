import getopt
import os
import sys
import traceback
import logging
from raidionicsseg.fit import run_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(argv):
    config_filename = None
    try:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        opts, args = getopt.getopt(argv, "h:c:", ["Config="])
    except getopt.GetoptError:
        print('usage: main.py --Config <config_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: main.py --Config <config_file>')
            sys.exit()
        elif opt in ("-c", "--Config"):
            config_filename = arg

    if not config_filename or not os.path.exists(config_filename):
        print('usage: main.py --Config <config_file>')
        sys.exit()

    try:
        run_model(config_filename=config_filename)
    except Exception as e:
        logging.error('{}'.format(traceback.format_exc()))


if __name__ == "__main__":
    main(sys.argv[1:])

