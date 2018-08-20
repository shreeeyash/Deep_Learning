# Face Recognition
I have written here code for simple face recognition system, which identify the face of the person if he is already in dataset!

## About Code
Mainly two libraries were used - OpenCV and face_recognition. face_recognition Adam Geitgey already contains pretrained model which can convert face image to 128d vector encoding. After knowing these encodings, we can use some distance function (sqaured L2 distance is proposed in FaceNet paper) to check whether the face of person in given image matches to any entries in dataset.
Here I have used the functions provided by face_recognition library to find encodings and distance.

## Dataset
As we already have pretrained model, we dont need millions of images to train *Inception Model*, we only need *some(~20 images)* images of person we want to recognize. Dataset can be created using webcam for which i have provided the code in [create_dataset_webcam.py](https://github.com/Shreeyash-iitr/Deep_Learning/blob/master/Face_Recognition/create_dataset_webcam.py) file. Other option is to manually download images from web.
## Result
I used 7 different person-faces in my dataset,along with my face images. As a test image, I am passing my own image to let the computer guess *who is this?*
