import csv
import sys
from multiprocessing import Pool, TimeoutError
import cv2
import os
from scipy.io import savemat
import numpy as np


def extract_batch(i):
    global BATCH_SIZE, files, calc_rows
    for f in range(len(files[i * BATCH_SIZE: (i + 1) * BATCH_SIZE])):

        idx = i * BATCH_SIZE + f
        csv_idx = calc_rows[idx]
        upload_folder = 'set' + str(csv_idx // 500 + 1) + '_upload'
        output_folder = os.path.join(testrt, testname, upload_folder + '_EXT_MAT')
        if not os.path.exists(output_folder):
            try:
                os.makedirs(output_folder)
            except FileExistsError:
                print('mat folder exists')

        vidpath = os.path.join(testrt, testname, upload_folder, files[idx][1])
        matpath = os.path.join(output_folder, files[idx][1].replace('mp4', 'mat'))
        if os.path.exists(matpath):
            print(f'file {matpath} already there')
            continue
        vidcap = cv2.VideoCapture(vidpath)
        totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        images = np.zeros((frameHeight, frameWidth, totalFrames), dtype=np.uint8)
        index = 0
        while index < totalFrames:
            success, image = vidcap.read()
            if success:
                img = 0.2989 * image[:, :, 2] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 0]
                img = img.astype(np.uint8)

                images[:, :, index] = img
                index += 1

        print(matpath)
        savemat(matpath, {'frames': images})


def extract_file(vidpath):
    matpath = vidpath.replace('mp4', 'mat')
    if os.path.exists(matpath):
        print(f'file {matpath} already there')
        return
    vidcap = cv2.VideoCapture(vidpath)
    totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameWidth = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    images = np.zeros((frameHeight, frameWidth, totalFrames), dtype=np.uint8)
    index = 0
    while index < totalFrames:
        success, image = vidcap.read()
        if success:
            img = 0.2989 * image[:, :, 2] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 0]
            img = img.astype(np.uint8)

            images[:, :, index] = img
            index += 1

    print(matpath)
    savemat(matpath, {'frames': images})


if __name__ == "__main__":
    print(sys.argv[1:])
    testrt = sys.argv[1]  # the root folder that contains all test folders
    testname = sys.argv[2]
    set_size = int(sys.argv[3])  # The number of mat files that is gonna be generated for matlab decoding
    set_num = int(sys.argv[4])  # which set of files
    Num_worker = int(sys.argv[5])
    BATCH_SIZE = int(sys.argv[6])

    csv_path = os.path.join(testrt, testname,
                            'wordlog_py_' + testname + '.csv')  # remember to change the csv name into this format
    with open(csv_path) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    calc_rows = list(range((set_num - 1) * set_size, set_num * set_size))
    files = [csv_reader[i] for i in calc_rows]

    with Pool(processes=Num_worker) as pool:
        for i in pool.imap_unordered(extract_batch, range(len(files) // BATCH_SIZE)):  # process all files
            pass

# extract_batch(0)