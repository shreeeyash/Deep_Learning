# import the necessary packages

from imutils import paths
import face_recognition
import pickle
import cv2
import os

# grab the paths to the input images in our dataset
imagepaths = list(paths.list_images('/home/shreeyash/PycharmProjects/ML/Face_Recognition/dataset'))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagepath) in enumerate(imagepaths):
    print(str(imagepath))
    # extract the person name from the image path
    name = imagepath.split(os.path.sep)[-2]
    # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
    img = cv2.imread(str(imagepath))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb_img)
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb_img, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("saving...")
data = {'encodings': knownEncodings,"names":knownNames}
f = open("/home/shreeyash/PycharmProjects/ML/Face_Recognition/encodings.pickle", "wb")
pickle.dump(data, f)
f.close()