import os
import pandas as pd
import csv
from operator import itemgetter

IMAGE_DATABASE_LOCAL_PATH = "/Users/niloleart/PycharmProjects/mini_dataset"
IMAGE_DATABASE_REMOTE_PATH = "/home/niloleart/fbm_mntdir/database/UPC - IMAGENES MARATO - ANONIMIZADAS"

ANNOTATIONS_CSV_LOCAL_FILE = "/Users/niloleart/PycharmProjects/mini_labels.csv"
ANNOTATIONS_CSV_REMOTE_FILE = "/home/niloleart/full_dataset.csv"

IMAGES_AND_LABELS_FILE_LOCAL_PATH = "/Users/niloleart/PycharmProjects/test.csv"
IMAGES_AND_LABELS_FILE_REMOTE_PATH = "home/niloleart/images_paths_and_labels.csv"

IMAGE_TYPE_MACULAR_CUBE = "macular"

CSV_DELIMITER = ','

# Add image type to get csv (angiography 3x3, profunda, color), something to do with L-R eyes too?
IMAGE_TYPE = "//TODO"


def is_image_file(filename):
    return filename.endswith(
        '.jpeg') or filename.endswith('.jpg') or filename.endswith('.png')


# or filename.endswith('.bmp')
# or filename.endswith('.tif')

def get_path_to_write_csv(local_mode):
    if local_mode:
        return IMAGES_AND_LABELS_FILE_LOCAL_PATH
    else:
        return IMAGES_AND_LABELS_FILE_REMOTE_PATH


def get_writer(local_mode):
    f = open(get_path_to_write_csv(local_mode), 'w')
    writer = csv.writer(f)
    return f, writer


def write_csv(labels, left_eye_paths, right_eyes_paths, folder, local_mode):
    try:
        f, writer = get_writer(local_mode)

        for index, label in enumerate(labels):
            writer.writerow([left_eye_paths[index], right_eyes_paths[index], label, folder[index]])
        f.close()
        print("Successfully writen", len(labels), "labels and paths into", get_path_to_write_csv(local_mode))
    except Exception as e:
        print("There has been an error writing csv!")


def get_eye(filename):
    if filename.__contains__("OD"):
        return "R"
    else:
        return "L"


def get_folder_name(path):
    return path.split('/')[-1]


def get_eyes_paths(files, root_path):
    left_eye = ""
    right_eye = ""
    for match in files:
        if IMAGE_TYPE_MACULAR_CUBE.upper() in match.upper():
            if match.__contains__('OD'):
                right_eye = os.path.join(root_path, match)
            else:
                left_eye = os.path.join(root_path, match)

    return left_eye, right_eye


def get_annotations_path(local_mode):
    if local_mode:
        return ANNOTATIONS_CSV_LOCAL_FILE
    else:
        return ANNOTATIONS_CSV_REMOTE_FILE


def set_images_database_path(local_mode):
    if local_mode:
        return IMAGE_DATABASE_LOCAL_PATH
    else:
        return IMAGE_DATABASE_REMOTE_PATH


class CreateCSV:
    @staticmethod
    def create_CSV(mode):
        in_labels = pd.read_csv((open(get_annotations_path(mode))), delimiter=CSV_DELIMITER)
        label_list = []
        left_eyes_paths = []
        right_eyes_paths = []
        dir_list = []

        database_path = set_images_database_path(mode)

        for dirIdx, folder in enumerate(sorted(os.listdir(database_path))):
            for _, _, files in os.walk(os.path.join(database_path, folder)):
                left_eye, right_eye = get_eyes_paths(files, os.path.join(database_path, folder))
                left_eyes_paths.append(left_eye)
                right_eyes_paths.append(right_eye)
                label_list.append(in_labels.values[dirIdx - 1][2])
                dir_list.append(get_folder_name(os.path.join(database_path, folder)))

        write_csv(label_list, left_eyes_paths, right_eyes_paths, dir_list, mode)

        return get_path_to_write_csv(mode)
