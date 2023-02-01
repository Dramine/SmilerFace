import os
import dlib
import numpy as np
from imutils import face_utils
import operator


def get_face_infos(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('%s/shape_predictor_68_face_landmarks.dat' % os.getcwd())

    width_image = image.shape[0]
    height_image = image.shape[1]
    base_coords = np.array([[0, 0], [height_image - 1, 0], [height_image - 1, width_image - 1], [0, width_image - 1]])

    det = detector(image, 1)[0]
    face_area = face_utils.rect_to_bb(det)
    face_area = np.array([face_area[:2], tuple(map(operator.add, face_area[:2], face_area[2:]))])
    coords = face_utils.shape_to_np(predictor(image, det))

    return face_area, base_coords, coords
