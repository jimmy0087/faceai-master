import os
import tensorflow as tf
import numpy as np
import requests
import tarfile
import zipfile
import shutil
import six
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    file_name = destination.split('/')[-1]
    save_path = '/'.join((destination.split('/')[0:-1]))
    if os.path.exists(destination):
        return destination
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    print('*'+file_name + " pre-trained model will download at "+ destination)
    fname = os.path.join(save_path, file_name + '.tar.gz')

    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    try:
        save_response_content(response, fname)
    except:
        print("*download failed, please try again.")
        os.remove(fname)

    try:
        t = tarfile.open(fname)
        t.extractall(path=save_path)
    except:
        print("*untar failed, please try again.")
        shutil.rmtree(destination)

    print("*download succeed !")

    return destination

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)



if __name__ == "__main__":
    file_id = '1-diIoodSWEVLdtcMJsftWhgaQF6J5a1I'
    destination = 'F:/postgraduate/program/detection/faceai-master/data/mnist.zip'
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=destination,
                                        unzip=False)
    download_file_from_google_drive(file_id, destination)
