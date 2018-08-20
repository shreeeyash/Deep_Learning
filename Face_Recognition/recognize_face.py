import face_recognition
import pickle
import cv2
import numpy

# load the known faces and embeddings
data = pickle.loads(open('/home/shreeyash/PycharmProjects/ML/Face_Recognition/encodings.pickle',"rb").read())

# load the input image and convert it from BGR to RGB
image = cv2.imread('/home/shreeyash/PycharmProjects/ML/Face_Recognition/test_img.JPG')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
box = face_recognition.face_locations(rgb)
encoding = face_recognition.face_encodings(rgb, box)
(y1, x2, y2, x1) = box[0]

# attempt to match each face in the input image to our known encodings
matches = face_recognition.compare_faces(numpy.array(data["encodings"]), numpy.array(encoding))


"""  ===== print(matches)====  
[False, False, False, False, False, False, False, False, False, False, False,
 False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False, 
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False,
False, False, False, False, False, False, False, False, False, False, False, 
False, False, False, False, True, True, True, True, True, True, True, True, True,
True, True, True, True, True, True, True, True, True, True, True, True, True, True,
True, True, True, True, True, True, True, True, True, True, True, True, True]

"""


# find the indexes of all matched faces then initialize a
# dictionary to count the total number of times each face
# was matched
matchedIdxs = [i for (i,b) in enumerate(matches) if b]
counts = {}

# lop over the matched indexes and maintain a count for each recognized face face
for i in matchedIdxs:
    name = data["names"][i]
    counts[name] = counts.get(name, 0) +1

# determine the recognized face with the largest number of votes
name = max(counts, key = counts.get)

cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
cv2.putText(image, name, (x1, y1-10),cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0),2)
cv2.imshow("output", image)
cv2.imwrite("/home/shreeyash/PycharmProjects/ML/Face_Recognition/output.png",image)
cv2.waitKey()
