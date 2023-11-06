# two-phase-selection

Code of the paper "A Two-Phase Recall-and-Select Framework for Fast Model Selection". Two phases refers to coarse recall and fine selection.

## coarse recall

The coarse recall is the first part of our model selection and the related files are in /coarse_recall. The main codes are in /coarse_recall/CV_carse_recall and /coarse_recall/NLP_coarse_recall, representing our experiment in natural language process and computer vision. The files in two folders are similar and they are:

- datasets.txt 

This file contains names of different datasets in each row.

- matrix.txt

This file contains an accuracy matrix for model clustering. The first row contains names of different models separated by '\t'.  The following lines, also separated by '\t' in each item, contain the accuracy of different models. This matrix is got off-line.

- leep_score_multirc.txt

This file contains the LEEP score of dataset MultiRC. The first line should match the first line of matrix.txt. The second line contains different models' LEEP score.

- model_clustering.py

This file identifies related class and methods that accomplish model clustering. One can get both hierarchical clustering and K-means clustering result in this file.

- coarse_recall_v2.py

This file calculates the coarse_recall result. This file instantiates the model clustering class and calls the hierarchical clustering method.

There are also other folders in /coarse_recall. The folder /coarse_recall/leep contains the code that calculate the LEEP score. The folder /coarse_recall/sbert contains the files and codes that calculate the text similarity according to the model description.
