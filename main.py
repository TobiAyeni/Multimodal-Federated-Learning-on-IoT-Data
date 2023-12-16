import configparser
import argparse
import logging
import os
import warnings
import torch
import matplotlib.pyplot as plt
from fl import FL
from server import Server
def read_config():
    config = configparser.ConfigParser()
    config.read('C:/Users/Ayeni/iotdi22-mmfl/config/opp/dccae/A0_B0_AB30_label_AB_test_B')
    return config

#print(torch.__version__)



config = read_config()
fl = FL(config)
fl.start()
fl.draw_graph()

# Per Class Accuracy
''' 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils import load_data
from sever import train_classifier

# Generate data
data_train, data_test = load_data(self.config)
X = data_train
y = data_test

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier 
classifierA = train_classifier() # Modality A
classifierB = train_classifier() # Modality B
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_predA = classifierA.predict(X_test)
y_predB = classifierB(X_test)

# Calculate overall accuracy
overall_accuracyA = accuracy_score(y_test, y_predA)
overall_accuracyB = accuracy_score(y_test,y_predB)
# Calculate per-class accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

# Print results
print("Overall Accuracy:", overall_accuracy)
print("Per-class Accuracy:")
for i, acc in enumerate(per_class_accuracy):
    print(f"Class {i}: {acc:.2f}")


'''
