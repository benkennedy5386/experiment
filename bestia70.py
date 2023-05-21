import os
import runpy
import time

files = get_subs('D:/models/research/object_detection/server/siding/')

print("Updated directory to ", files)

for file in files:
    PATH_TO_TEST_IMAGES_DIR = 'D:/models/research/object_detection/server/siding/'+file+'/'
    PATH_TO_TEST_INFO = 'D:/models/research/object_detection/server/siding/'+file+'.txt'
        #rename the photos
    def rename():
        i = 1
        for name in os.listdir(PATH_TO_TEST_IMAGES_DIR):
            dst = PATH_TO_TEST_IMAGES_DIR + "image" + str(i) + ".jpg"
            src = PATH_TO_TEST_IMAGES_DIR + name


            os.rename(src, dst)
            i += 1
    rename()
    print('renaming ok')

    #run object detection
    magic()
    print('successfully processed ', file)



print("Sleeping for 60 seconds")
time.sleep(60)
