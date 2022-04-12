# Pneumothorax Classifier
[![deploy-frontend](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-frontend.yml/badge.svg)](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-frontend.yml)
[![deploy-model](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-model.yml/badge.svg)](https://github.com/cianmawhinney/microsoft-pneumothorax-classifier/actions/workflows/deploy-model.yml)

A machine learning application to classify chest X-rays for detecting pneumothorax, developed during Trinity College Dublin's Software Engineering Project module, with mentorship from engineers at Microsoft.


Pneumothorax is a serious medical condition where air leaks into the space between the lung and the chest wall. Diagnosis of the condition is currently performed through the use of a chest X-ray, a difficult process requiring specialist expertise to perform and interpret. Due to the fact that pneumothorax can be life threatening, a fast, accurate diagnosis is desirable, though often wait times can be long due to the workload on staff, or in lesser developed locations staff may not be available at all.


After speaking with staff at Midland Regional Hospital Tullamore, it became clear Microsoft, to our client and mentors, that interpreting the X-rays to give a diagnosis would be a good candidate for automation, potentially allowing for faster and more accurate diagnoses.


# Front end UI
The front end web interface was created using HTML, JS, CSS and the Bootstrap 5 framework.
Multiple images can be uploaded and the results displayed to the end user so a diagnosis can be made.

Primary objective of design was to create a clean, accessible interface. The main goal of this project was to save medical professionals time which we emulated in the front-end.

## Screenshots
### Homepage Screenshot
![Homepage Screenshot](docs/images/frontend-home.png)

### Results Screenshot
![Results Screenshot](docs/images/frontend-home.png)

# ML Model

## Dataset
The page for the dataset we used can be found here: https://www.kaggle.com/vbookshelf/pneumothorax-chest-xray-images-and-masks
It has been changed from the original dataset used in that competition to make it easier to work with. The images and masks have been converted to pngs instead
of dcim and rngs. 

A second version of the dataset has been created in azure where the masks have been removed for our first round of testing, and also the images have been moved into train and tes subfolders. To allow us to more easily access the relavent images for each step.

### Stats
Number of files: 12047  
| # Training Images | 10675 |
|-------------------|-------|
| Positive          | 2379  |
| Negative          | 8296  |

| # Testing Images | 1372 |
|------------------|------|
| Positive         | 290  |
| Negative         | 1082 |

#### Model Results
| # Name                                   | # Loss | # Accuracy | # Recall          | # Precision |
|------------------------------------------|--------|------------|-------------------|-------------|
| VGG No Augmentation Fine Tuning          | 0.3922 | 0.8393     | 0.6512            | 0.608       |
| VGG with augmentation                    | 0.4554 | 0.7902     | 0? (not sure why) | 0           |
| VGG no augmentation                      | 0.3866 | 0.84       | 0.3936            | 0.7167      |
| VGG with augmentation and fine tuning    | 0.3629 | 0.84       | 0.4036            | 0.7019      |
| VGG with trans and zoom and fine tuninng | 0.3752 | 0.8608     | 0.5               | 0.7592      |
| EfficientNet with augmentation           | 1.1644 | 0.7909     | N/A               | N/A         |
| EfficientNet no augmentation             | 0.509  | 0.8229     | N/A               | N/A         |

# Infrastructure
The infrastructure is hosted in Azure, using ML Studio. Deployments are automated using GitHub Actions so that when changes are made to the model or user interface, they are

## Architecture Diagram
![Architecture Diagram](docs/images/Architecture%20Diagram.drawio.svg)

# Authors
 - [Cian Mawhinney](https://github.com/cianmawhinney)
 - [Dermot O'Brien](https://github.com/mangledbottles)
 - [Emmet McDonald](https://github.com/EmmetMcD)
 - [Kenneth Harmon](https://github.com/KennethHarmon)
 - [Elliot Lyons](https://github.com/elliot-lyons)
 - [Jacob Sharpe](https://github.com/j-wilsons)
 - [Aj Pakulyte](https://github.com/pakulyta)
