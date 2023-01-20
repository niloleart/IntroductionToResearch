import os

import numpy as np
import pandas as pd
import csv

IMAGE_DATABASE_LOCAL_PATH = "/Users/niloleart/PycharmProjects/mini_dataset"
IMAGE_DATABASE_REMOTE_PATH = "/home/niloleart/fbm_mntdir/database/UPC - IMAGENES MARATO - ANONIMIZADAS"

# ANNOTATIONS_CSV_LOCAL_FILE = "/Users/niloleart/PycharmProjects/mini_labels.csv"  # TODO: uncomment
ANNOTATIONS_CSV_LOCAL_FILE = "/Users/niloleart/PycharmProjects/mini_double_labels.csv"
# ANNOTATIONS_CSV_LOCAL_FILE = "/Users/niloleart/PycharmProjects/mini_labels_imbalance.csv"
# ANNOTATIONS_CSV_REMOTE_FILE = "/home/niloleart/full_dataset.csv"
ANNOTATIONS_CSV_REMOTE_FILE = "/home/niloleart/labels_no_treatment.csv"

IMAGES_AND_LABELS_FILE_LOCAL_PATH = "/Users/niloleart/PycharmProjects/test.csv"
IMAGES_AND_LABELS_FILE_REMOTE_PATH = "/home/niloleart/images_paths_and_labels.csv"

images_type = {
    'macular': ['macular'],
    'color': ['color'],
    'angiography_3x3_profundidad': ['angiography', '3x3', 'profundidad'],
    'angiography_3x3_superficial': ['angiography', '3x3', 'superficial'],
    'angiography_6x6_profundidad': ['angiography 6x6', 'profundidad'],
    'angiography_6x6_superficial': ['angiography 6x6', 'superficial'],
}

CSV_DELIMITER = ','  # For remote
# CSV_DELIMITER = ';'  # For local


# DM: Diabetic Macular Edema -> 0 vs 1-5
# DR: Diabetic Retinopathy -> 1 vs 2-5 (0 excluded)
# R-DR: Referable Diabetic Retinopathy -> 1-2 vs 3-5 (0 excluded)
experiments = {
    'DM_diagnosis': 0,
    'DR_diagnosis': 1,
    'R-DR_diagnosis': 2
}

experiment = experiments['DM_diagnosis']



def is_image_file(filename):
    return filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith(
        '.bmp') or filename.endswith('.tif')


def get_path_to_write_csv(local_mode):
    if local_mode:
        return IMAGES_AND_LABELS_FILE_LOCAL_PATH
    else:
        return IMAGES_AND_LABELS_FILE_REMOTE_PATH


def get_writer(local_mode):
    f = open(get_path_to_write_csv(local_mode), 'w')
    writer = csv.writer(f)

    return f, writer


def write_csv(labels_left, labels_right, macular_left_eye_paths, macular_right_eyes_paths,
              color_left_eyes_paths, color_right_eyes_paths,
              octa_3x3_sup_left, octa_3x3_sup_right,
              octa_3x3_deep_left, octa_3x3_deep_right,
              octa_6x6_sup_left, octa_6x6_sup_right,
              octa_6x6_deep_left, octa_6x6_deep_right,
              folder,
              local_mode):
    header = ['left_macular', 'right_macular',
              'left_color', 'right_color',
              'left_octa_3x3_sup', 'right_octa_3x3_sup',
              'left_octa_3x3_deep', 'right_octa_3x3_deep',
              'left_octa_6x6_sup', 'right_octa_6x6_sup',
              'left_octa_6x6_deep', 'right_octa_6x6_deep',
              'folder', 'label_left', 'label_right']
    try:
        f, writer = get_writer(local_mode)
        writer.writerow(header)
        for index, label in enumerate(labels_right):
            if not experiment == experiments['R-DR_diagnosis']:
                if label != 0:
                    writer.writerow([macular_left_eye_paths[index], macular_right_eyes_paths[index],
                                     color_left_eyes_paths[index], color_right_eyes_paths[index],
                                     octa_3x3_sup_left[index], octa_3x3_sup_right[index],
                                     octa_3x3_deep_left[index], octa_3x3_deep_right[index],
                                     octa_6x6_sup_left[index], octa_6x6_sup_right[index],
                                     octa_6x6_deep_left[index], octa_6x6_deep_right[index],
                                     folder[index],
                                     get_label(labels_left[index]), get_label(labels_right[index])])
            else:
                writer.writerow([macular_left_eye_paths[index], macular_right_eyes_paths[index],
                                 color_left_eyes_paths[index], color_right_eyes_paths[index],
                                 octa_3x3_sup_left[index], octa_3x3_sup_right[index],
                                 octa_3x3_deep_left[index], octa_3x3_deep_right[index],
                                 octa_6x6_sup_left[index], octa_6x6_sup_right[index],
                                 octa_6x6_deep_left[index], octa_6x6_deep_right[index],
                                 folder[index],
                                 get_label(labels_left[index]), get_label(labels_right[index])])
        f.close()
        print("Successfully writen labels and paths into", get_path_to_write_csv(local_mode))
    except Exception as e:
        print("There has been an error writing csv!")

    csv.DictWriter(open(get_path_to_write_csv(local_mode), 'a'), header)


def get_label(label):
    if experiment == experiments['DM_diagnosis']:
        if label == 0:
            return 0
        elif label > 0:
            return 1
    elif experiment == experiments['DR_diagnosis']:
        if label == 1:
            return 0
        elif label > 1:
            return 1
    elif experiment == experiments['R-DR_diagnosis']:
        if label == 1 or label == 2:
            return 0
        elif label > 2:
            return 1
    else:
        print("ERROR: choose correct experiment!")
        exit()


def get_folder_name(path):
    return path.split('/')[-1]


def get_condition(match, keywords):
    if len(keywords) > 1:
        return ((keywords[0].upper() and keywords[2].upper()) in match.upper()) and (
                match.split(' ')[1].upper() == keywords[1].upper)
    else:
        return keywords[0].upper() in match.upper()


def get_images_paths(files, root_path, keywords):
    left_eye = ""
    right_eye = ""
    for match in files:
        if all(x.upper() in match.upper() for x in keywords):
            if match.__contains__('OD') or match.__contains__('R'):
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


def label_to_int(label: float):
    if np.isnan(label):
        return label
    else:
        return label.astype(int)


class CreateCSV:
    @staticmethod
    def create_CSV(mode):
        in_labels = pd.read_csv((open(get_annotations_path(mode))), delimiter=CSV_DELIMITER)
        label_list_right = []
        label_list_left = []
        dir_list = []

        macular_left_eyes_paths = []
        macular_right_eyes_paths = []
        color_left_eyes_paths = []
        color_right_eyes_paths = []
        angiography_3x3_profundidad_left_paths = []
        angiography_3x3_profundidad_right_paths = []
        angiography_3x3_superficial_left_paths = []
        angiography_3x3_superficial_right_paths = []
        angiography_6x6_superficial_left_paths = []
        angiography_6x6_superficial_right_paths = []
        angiography_6x6_profundidad_left_paths = []
        angiography_6x6_profundidad_right_paths = []

        database_path = set_images_database_path(mode)

        for idx, folder in enumerate(sorted(os.listdir(database_path))):
            for _, _, files in os.walk(os.path.join(database_path, folder)):
                root_path = os.path.join(database_path, folder)
                if idx < len(in_labels):
                    macular_left_eye, macular_right_eye = get_images_paths(files, root_path, images_type['macular'])
                    macular_left_eyes_paths.append(macular_left_eye)
                    macular_right_eyes_paths.append(macular_right_eye)

                    color_left_eye, color_right_eye = get_images_paths(files, root_path, images_type['color'])
                    color_left_eyes_paths.append(color_left_eye)
                    color_right_eyes_paths.append(color_right_eye)

                    # 3x3 Superficial
                    angiography_3x3_profundidad_left_eye, angiography_3x3_profundidad_right_eye = get_images_paths(
                        files,
                        root_path,
                        images_type[
                            'angiography_3x3_profundidad'])
                    angiography_3x3_profundidad_left_paths.append(angiography_3x3_profundidad_left_eye)
                    angiography_3x3_profundidad_right_paths.append(angiography_3x3_profundidad_right_eye)

                    # 3x3 Profundidad
                    angiography_3x3_superficial_left_eye, angiography_3x3_superficial_right_eye = get_images_paths(
                        files,
                        root_path,
                        images_type[
                            'angiography_3x3_superficial'])
                    angiography_3x3_superficial_left_paths.append(angiography_3x3_superficial_left_eye)
                    angiography_3x3_superficial_right_paths.append(angiography_3x3_superficial_right_eye)

                    # 6x6 profundidad
                    angiography_6x6_profundidad_left_eye, angiography_6x6_profundidad_right_eye = get_images_paths(
                        files,
                        root_path,
                        images_type[
                            'angiography_6x6_profundidad'])
                    angiography_6x6_profundidad_left_paths.append(angiography_6x6_profundidad_left_eye)
                    angiography_6x6_profundidad_right_paths.append(angiography_6x6_profundidad_right_eye)

                    # 6x6 superficial
                    angiography_6x6_superficial_left_eye, angiography_6x6_superficial_right_eye = get_images_paths(
                        files,
                        root_path,
                        images_type[
                            'angiography_6x6_superficial'])
                    angiography_6x6_superficial_left_paths.append(angiography_6x6_superficial_left_eye)
                    angiography_6x6_superficial_right_paths.append(angiography_6x6_superficial_right_eye)
                    label_list_right.append(label_to_int(in_labels.values[idx][1]))
                    label_list_left.append(label_to_int(in_labels.values[idx][2]))
                    dir_list.append(get_folder_name(os.path.join(database_path, folder)))

        write_csv(label_list_left, label_list_right,
                  macular_left_eyes_paths, macular_right_eyes_paths,
                  color_left_eyes_paths, color_right_eyes_paths,
                  angiography_3x3_superficial_left_paths, angiography_3x3_superficial_right_paths,
                  angiography_3x3_profundidad_left_paths, angiography_3x3_profundidad_right_paths,
                  angiography_6x6_superficial_left_paths, angiography_6x6_superficial_right_paths,
                  angiography_6x6_profundidad_left_paths, angiography_6x6_profundidad_right_paths,
                  dir_list, mode)

        return get_path_to_write_csv(mode)
