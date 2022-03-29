# Pneumothorax Classifier
[![deploy-frontend](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-frontend.yml/badge.svg)](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-frontend.yml)
[![deploy-model](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-model.yml/badge.svg)](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-model.yml)

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

#### Model Results
| # Name                          | # Loss | # Accuracy | # Recall | # Precision |
| - | - | - | - | - |
| VGG No Augmentation Fine Tuning | 0.3922 | 0.8393 | 0.6512 | 0.608 |
| VGG with augmentation           | 0.4554 | 0.7902 | 0? (not sure why) | 0 |
| VGG no augmentation             | 0.3866 | 0.84   | 0.3936 | 0.7167 |
| VGG with augmentation and fine tuning | 0.3629 | 0.84 | 0.4036 | 0.7019 | 
| VGG with trans and zoom and fine tuninng | 0.3752 | 0.8608 | 0.5 | 0.7592 |
| EfficientNet with augmentation | 1.1644 | 0.7909 | N/A | N/A | 
| EfficientNet no augmentation | 0.509 | 0.8229 | N/A | N/A |
