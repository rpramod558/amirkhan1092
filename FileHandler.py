import base64
import os
from zipfile import ZipFile
import urllib.request


def compressZipFile(filepath, directory):

    # calling function to get all file paths in the directory
    file_paths = get_all_file_paths(directory)

    # printing the list of all files to be zipped
    # print('Following files will be zipped:')
    # for file_name in file_paths:
    #     print(file_name)

    # writing files to a zipfile
    with ZipFile(filepath, 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            zip.write(file, os.path.basename(file))


def loadFile(filename):
    with open(filename, "rb") as fh:
        out_data = base64.b64encode(fh.read())

    return out_data


def downloadUrlImage(url, file_path):
    urllib.request.urlretrieve(url, file_path)


def getAllUrlImage(urls, folder):
    for url in urls:
        # try:
        downloadUrlImage(
            url,
            os.path.join(folder,
                         os.path.basename(urllib.request.urlparse(url).path)))
        # except:
        #     pass


def get_all_file_paths(directory):

    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths
    return file_paths
