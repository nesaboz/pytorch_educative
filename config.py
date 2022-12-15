import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display
from tensorboard import manager


def tensorboard_cleanup():
    info_dir = manager._get_info_dir()
    shutil.rmtree(info_dir)


FILEPATHS = {
    'transfer_learning': ['pytorched/step_by_step.py', 'data_generation/rps.py'],
}

try:
    import google.colab
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

IS_LOCAL = not IS_COLAB


def download_to_colab(project_name, branch='master'):
    root_url = 'https://github.com/nesaboz/pytorched/tree/{}'.format(branch)

    for filepath in FILEPATHS:
        folder, filename = os.path.split(filepath)
        os.makedirs(folder, exist_ok=True)
        url = os.path.join(root_url, filepath)
        r = requests.get(url, allow_redirects=True)
        open(filepath, 'wb').write(r.content)


def config_project(name, branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(name, branch)
        print('Finished!')
