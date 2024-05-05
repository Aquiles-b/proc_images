import urllib.request
import zipfile
import os

link = 'https://zenodo.org/records/10219797/files/macroscopic0.zip?download=1'
file_name = 'macroscopic0.zip'

print(f'Downloading {file_name}...')
urllib.request.urlretrieve(link, file_name)

with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall('.')

os.remove(file_name)

print('Completed!')
