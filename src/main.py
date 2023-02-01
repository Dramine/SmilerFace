import random
from skimage import io
import os
import sys
import cv2
import numpy as np
import detector
import warper
import neural_network


def get_random_images_couple():
    def to_str(num):
        if num < 10:
            num = "0" + str(num)
        else:
            num = str(num)
        return num

    all_i = range(1, 50)
    all_j = [1, 14]
    i = random.choice(all_i)
    j = random.choice(all_j)

    neutral_image = io.imread('%s/../data/img/M-0%s-%s.bmp' % (os.getcwd(), to_str(i), to_str(j)))
    smile_image = io.imread('%s/../data/img/M-0%s-%s.bmp' % (os.getcwd(), to_str(i), to_str(j + 1)))

    return neutral_image, smile_image


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Veuillez précisez un mode ! `--train` ou `--predict`")
        exit()

    elif sys.argv[1] == "--train":
        netMan = neural_network.NetworkTrainer()
        netMan.train()

    elif sys.argv[1] == "--predict":
        neutral_image, smile_image = get_random_images_couple()
        neutral_face_area, neutral_base_coords, neutral_coords = detector.get_face_infos(neutral_image)
        smile_face_area, smile_base_coords, smile_coords = detector.get_face_infos(smile_image)

        netMan = neural_network.NetworkPredictor()
        predicted_coords = netMan.predict(neutral_image)

        warping_src_coords = np.concatenate((neutral_base_coords, neutral_coords))
        warping_dest_coords = np.concatenate((neutral_base_coords, neutral_coords[:48], predicted_coords))
        warped_image = warper.warpRBF(neutral_image, warping_src_coords, warping_dest_coords)

        while True:
            neutral_im = cv2.cvtColor(neutral_image, cv2.COLOR_BGR2RGB)
            for coord in np.concatenate((neutral_base_coords, neutral_coords)):
                cv2.circle(neutral_im, tuple(coord), 2, (255, 0, 0), -1)
            for coord in predicted_coords:
                cv2.circle(neutral_im, tuple(coord), 2, (0, 255, 0), -1)

            smile_im = cv2.cvtColor(smile_image, cv2.COLOR_BGR2RGB)
            for coord in np.concatenate((smile_base_coords, smile_coords)):
                cv2.circle(smile_im, tuple(coord), 2, (255, 0, 0), -1)
            for coord in predicted_coords:
                cv2.circle(smile_im, tuple(coord), 2, (0, 255, 0), -1)

            warped_im = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)

            cv2.imshow("Neutral", neutral_im)
            cv2.imshow("Smile", smile_im)
            cv2.imshow("Warped", warped_im)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27=ESC
                break
        cv2.destroyAllWindows()

    else:
        print("Le mode précisez est incorrect !")
        print("Veuillez précisez un mode ! `--train` ou `--predict`")
