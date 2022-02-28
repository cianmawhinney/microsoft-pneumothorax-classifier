# Pneumothorax Classifier
A machine learning application to classify chest X-rays for detecting pneumothorax

# Dataset
The page for the dataset we used can be found here: https://www.kaggle.com/vbookshelf/pneumothorax-chest-xray-images-and-masks
It has been changed from the original dataset used in that competition to make it easier to work with. The images and masks have been converted to pngs instead
of dcim and rngs. 

A second version of the dataset has been created in azure where the masks have been removed for our first round of testing, and also the images have been moved into train and tes subfolders. To allow us to more easily access the relavent images for each step.

### Stats
Number of files: 12047  
| # Training Images | 10675 |  
| - | - |
| Positive | 2379 |
| Negative | 8296 |

| # Testing Images | 1372 |  
| - | - |
| Positive | 290 |
| Negative | 1082 |
