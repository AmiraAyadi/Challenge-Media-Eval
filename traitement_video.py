# -*-coding:Utf-8 -*

# import ffmpy
# ff = ffmpy.FFmpeg(
#     inputs={'inputs.flv.ogv': None},
#     outputs={'output.avi': None}
# )
# ff.run()

import numpy as np
import cv2
import os
import json

path_r = "Dataset/videos/"
path_w = "created_data/videos_images/"

types_of_videos = os.listdir(path_r)

for type_v in types_of_videos:
    videos = os.listdir(path_r+type_v)

    if not os.path.exists(path_w+type_v):
        os.makedirs(path_w+type_v)

    for video in videos:
        
        print(video)

        if not os.path.exists(path_w+type_v+"/"+video):
            os.makedirs(path_w+type_v+"/"+video)

        cap = cv2.VideoCapture(path_r+type_v + "/" + video)

        number_of_frame = int(cap.get(7))
        print(number_of_frame)

        time = int(np.around(number_of_frame/cap.get(5)))

        if time < 150:
            nb_image = 15
        else:
            nb_image = 30

        splits_frame = [int(np.around((i)*(number_of_frame/nb_image))) for i in range(min(nb_image+1, 100000))]

        choosen_ones = [np.random.choice(np.arange(splits_frame[i], splits_frame[i+1]), 1) for i in range(min(nb_image, 100000))]

        info = {}

        j = 0

        for i in range(min(number_of_frame, 100000)):

            ret, frame = cap.read()

            if i in choosen_ones:
                cv2.imwrite(path_w+type_v+"/"+video+"/image_{}.png".format(i), frame)
                info["index"] = j
                j += 1

                info["time"] = i/cap.get(5)

                info["name"] = path_w+type_v+"/"+video+"/image_{}.png".format(i)
            
            with open(path_w+type_v+"/"+video+"/info.json", "w") as file:
                json.dump(info, file)

        
            




