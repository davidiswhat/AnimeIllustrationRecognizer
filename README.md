# AnimeIllustrationRecognizer
AnimeIllustrationRecognizer is an application to recognize Anime characters and tags (labels) for a given illustration by using an Attention Model.

# Installation Instructions
We apologize as we do not have a convenient way to install all the package dependencies needed nor do we have them all listed. A major one used was Tensorflow. 

## Prerequisites
- Due to us reaching the limit of GitHub's LFS free data plan, you must download the Attention Model from the following Google Drive: https://drive.google.com/file/d/1V6ztX0O3w4_w8k8otXrQjylX2_biAX3I/view?usp=sharing
- Then move the approximately 600MB file into AnimeFaceNotebooks/deepdanbooru_model/ 

Run MainWindow.py after installation

# Usage
- Note: Character names are shown and are treated as a tag
- Note: output.txt and output.jpg are generated from the most recent output
- Note: The two buttons, "Mask" and "Stylegan", currently do nothing and are hidden from users

## Main Window 
![](https://github.com/davidiswhat/AnimeIllustrationRecognizer/blob/main/Screenshots/MainWindow.PNG)
- Click the "Load a new image" button to open the add image window interface
- You can submit multiple images (*.png, *.jpg, *.jpeg) to the database at once
- Each submission to the database requires a unique submission name
- You only need to submit the file name (e.g. sub1.jpg) in the second textbox if it is a unique file name in the database; otherwise, you must also include the submission name in the first textbox
- Click the "Submit" button to view the image from the database and generate the tag outputs 

## Image Preprocessing
![](https://github.com/davidiswhat/AnimeIllustrationRecognizer/blob/main/Screenshots/ImagePreprocessing.PNG)
- The add image window interface includes checkboxes to crop the face after detection or to apply grayscale
- This image showcases an example of both of them being applied and how you would include the submission name in the first textbox

## Pixelate
![](https://github.com/davidiswhat/AnimeIllustrationRecognizer/blob/main/Screenshots/Pixelate.PNG)
- WARNING: This is a time-consuming process
- Clicking the "Pixelate" button will open a window where the user can input a tag (e.g. headphones) to pixelate
- The image will pixelate the relevant area(s) the model used to calculate the tag's confidence coefficient 

# Training Dataset
- The database of the Anime imageboard, Danbooru, was the source of the training data
- Danbooru contains Anime images that are labeled with the character name(s) and a variety of other community labeled tags
- SFW (Safe For Work) images for each of the tags (character names were treated as a tag) were fed to the Attention Model

# Machine Learning Code References
Our code directly referenced the code in the following repositories:
- https://github.com/halcy/AnimeFaceNotebooks
- https://github.com/halcy/DeepDanbooruActivationMaps
- https://github.com/nagadomi/lbpcascade_animeface

We give special thanks to them
