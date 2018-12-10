import os
import requests
import tarfile
import shutil


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    file_name = destination.split('/')[-1]
    save_path = '/'.join((destination.split('/')[0:-1]))
    if os.path.exists(destination):
        return destination
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    fname = os.path.join(save_path, file_name + '.tar.gz')
    try:
        print('*'+file_name + " pre-trained model will download at "+ destination)
        session = requests.Session()
        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)
        save_response_content(response, fname)
    except :
        print("*download failed, please try again.")
        if os.path.exists(fname):
            os.remove(fname)

    try:
        t = tarfile.open(fname)
        t.extractall(path=save_path)
        print("*download succeed !")
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        print("*untar failed, please try again.")
        if os.path.exists(destination):
            shutil.rmtree(destination)

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
