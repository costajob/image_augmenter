from logging import basicConfig, ERROR, getLogger


FILENAME = 'imgaug.log'
FORMAT = '[%(asctime)s #%(process)d] -- %(levelname)s : %(message)s'
basicConfig(filename=FILENAME, format=FORMAT, level=ERROR)
BASE = getLogger(__name__)
