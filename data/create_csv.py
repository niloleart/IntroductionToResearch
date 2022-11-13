import os
import pandas as pd
import csv
from operator import itemgetter

DATABASE_PATH = "/Users/niloleart/PycharmProjects/mini_dataset"
ANNOTATIONS_CSV_FILE = "/Users/niloleart/PycharmProjects/mini_labels.csv"
IMAGES_AND_LABELS_FILE_PATH = "/Users/niloleart/PycharmProjects/test.csv"

IMAGE_TYPE_MACULAR_CUBE = "macular"

CSV_DELIMITER = ','

# Add image type to get csv (angiography 3x3, profunda, color), something to do with L-R eyes too?
IMAGE_TYPE = "//TODO"


def is_image_file(filename):
    return filename.endswith(
        '.jpeg') or filename.endswith('.jpg') or filename.endswith('.png')


# or filename.endswith('.bmp')
# or filename.endswith('.tif')

def get_writer():
    f = open(IMAGES_AND_LABELS_FILE_PATH, 'w')
    writer = csv.writer(f)
    return f, writer


def write_csv(labels, left_eye_paths, right_eyes_paths,  folder, local_mode=True):
    try:
        f, writer = get_writer()

        for index, label in enumerate(labels):
            writer.writerow([left_eye_paths[index], right_eyes_paths[index], label, folder[index]])
        f.close()
        print("Successfully writen", len(labels), "labels and paths into", IMAGES_AND_LABELS_FILE_PATH)
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


class CreateCSV:
    @staticmethod
    def create_CSV():
        in_labels = pd.read_csv((open(ANNOTATIONS_CSV_FILE)), delimiter=CSV_DELIMITER)
        label_list = []
        left_eyes_paths = []
        right_eyes_paths = []
        dir_list = []

        for dirIdx, folder in enumerate(sorted(os.listdir(DATABASE_PATH))):
            for _, _, files in os.walk(os.path.join(DATABASE_PATH, folder)):
                left_eye, right_eye = get_eyes_paths(files, os.path.join(DATABASE_PATH, folder))
                left_eyes_paths.append(left_eye)
                right_eyes_paths.append(right_eye)
                label_list.append(in_labels.values[dirIdx - 1][2])
                dir_list.append(get_folder_name(os.path.join(DATABASE_PATH, folder)))

        write_csv(label_list, left_eyes_paths, right_eyes_paths, dir_list)

        return IMAGES_AND_LABELS_FILE_PATH
