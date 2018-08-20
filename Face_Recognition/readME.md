# Face Recognition
I have written here code for simple face recognition system, which identify the face of the person if he is already in dataset!

##About Code
Mainly two libraries were used - OpenCV and face_recognition. face_recognition Adam Geitgey already contains pretrained model which can convert face image to 128d vector encoding. After knowing these encodings, we can use some distance function (sqaured L2 distance is proposed in FaceNet paper) to check whether the face of person in given image matches to any entries in dataset.
Here I have used the functions provided by face_recognition library to find encodings and distance.
