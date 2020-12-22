# AnimeIllustrationRecognizer
AnimeIllustrationRecognizer is an application to recognize Anime characters and tags (labels) for a given illustration.

# Installation Instructions
We apologize as we do not have a convenient way to install all the package dependencies needed nor do we have them listed. 

A major one used was Tensorflow. 

# Usage

## Main Window 
![](https://github.com/davidiswhat/AnimeIllustrationRecognizer/blob/main/Screenshots/MainWindow.PNG)
- ddd
## Image Preprocessing
![](https://github.com/davidiswhat/AnimeIllustrationRecognizer/blob/main/Screenshots/ImagePreprocessing.PNG)
- ddd
## Pixelate
![](https://github.com/davidiswhat/AnimeIllustrationRecognizer/blob/main/Screenshots/Pixelate.PNG)
- WARNING: This is a time consuming process
- Clicking the Pixelate button will open a window where the user can input a tag to pixelate
- The image will pixelate the relevant area(s) the model used to calculate the confidence coefficient 
- Note: The other two buttons, Mask and Stylegan, currently do nothing and are actually hidden

# Machine Learning Code References
Our code directly referenced the work done in the following repositories:
- https://github.com/halcy/AnimeFaceNotebooks
- https://github.com/halcy/DeepDanbooruActivationMaps
- https://github.com/nagadomi/lbpcascade_animeface

We give special thanks to them
