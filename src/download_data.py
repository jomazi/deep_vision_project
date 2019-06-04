import os
import requests
import zipfile
import io


def download_data(dir='../data/'):
    '''
    :param dir: relative path to directory the data is loaded into; do not forget the backslash at the end
    :return: -
    '''

    # only load files if they do not already exist
    current_dir = os.path.dirname(os.path.realpath(__file__))
    load_dir = os.path.join(current_dir, dir)

    if not os.path.isdir(os.path.join(load_dir, 'LLD-icon')):
        print('data is getting downloaded ... might take a few minutes :)')
        r = requests.get('https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-icon_PKL.zip')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(load_dir)


if __name__ == '__main__':
    download_data()