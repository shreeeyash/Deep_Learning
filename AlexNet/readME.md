## PCA augmentation
""The second form of data augmentation consists of altering the intensities of the RGB channels in
training images. Specifically, we perform PCA on the set of RGB pixel values throughout the
ImageNet training set. To each training image, we add multiples of the found principal components,The second form of data augmentation consists of altering the intensities of the RGB channels in
training images. Specifically, we perform PCA on the set of RGB pixel values throughout the
ImageNet training set. To each training image, we add multiples of the found principal components,</br>
                          [p 1 , p 2 , p 3 ][α 1 λ 1 , α 2 λ 2 , α 3 λ 3 ] T</br>
where p i and λ i are ith eigenvector and eigenvalue of the 3 × 3 covariance matrix of RGB pixel
values, respectively, and α i is the aforementioned random variable.""</br>
<img src="output.png">
