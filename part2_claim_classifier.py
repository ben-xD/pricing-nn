import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from NeuralNet import NeuralNet
from statistics import mean # TODO - can this be imported?
import matplotlib.pyplot as plt

#METRICS
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class ClaimClassifier():

    def __init__(self,):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary. 
        """
        self.trained_model = None

    def _preprocessor(self, X_raw):
        """Data preprocessing function.

        This function prepares the features of the data for training,
        evaluation, and prediction.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A clean data set that is used for training and prediction.
        """
        # YOUR CODE HERE
        X_raw = X_raw.to_numpy()

        # Normalisation
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X_raw)

        # TODO - Save the min_max_scaler object

        # Return full dataset, normalised dataset without dropping the columns as numpy array
        return X_scaled


    def fit(self, X_raw, y_raw):
        """Classifier training function.

        Here you will implement the training function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded
        y_raw : ndarray (optional)
            A one dimensional array, this is the binary target variable

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)
        # YOUR CODE HERE

        # TODO - save these somewhere else: Hyperparameters
        hidden_size = 50  # model
        num_epochs = 2  # fit / train
        batch_size = 10  # fit / train
        learning_rate = 0.01  # fit / train
        weighting = 4  # fit / train

        num_train = int(X_raw.shape[0] * 0.8)

        train_data = X_clean[:num_train]
        train_labels = y_raw[:num_train]

        test_data = X_clean[num_train:]
        test_labels = y_raw[num_train:]

        # Convert from numpy to tensors for train data and corresponding labels
        # NB X_clean is already a numpy array
        x_train = torch.from_numpy(train_data)
        y_train = torch.from_numpy(train_labels.to_numpy())

        x_test = torch.from_numpy(test_data)
        y_test = torch.from_numpy(test_labels.to_numpy())

        # Training dataset
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Testing dataset
        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Input and Output
        num_inputs = train_data.shape[1]
        input_size = num_inputs
        output_size = 1

        # Create a model with hyperparameters
        model = NeuralNet(input_size, hidden_size, output_size)

        # Weight positive samples higher
        pos_weight = torch.ones([batch_size])
        pos_weight.fill_(weighting)

        # Loss criterion and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        #TODO - to delete
        accuracy_list = []
        epochs_list = []
        training_loss = []
        batch_loss = []
        batch_accuracy = []

        for epoch in range(num_epochs):
            for xb, yb in train_dl:

                # Forwards pass
                preds = model(xb.float())  # Why do I need to add float() here?
                loss = criterion(preds.flatten(), yb.float())

                # For calculating the average loss and accuracy
                batch_loss.append(loss.item())
                batch_accuracy.append(model.accuracy(preds, yb, batch_size))

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epochs_list.append(epoch)
            print(epoch)
            training_loss.append(mean(batch_loss))
            accuracy_list.append(mean(batch_accuracy))

        plt.plot(epochs_list, training_loss, 'g', label='Training loss')
        # plt.plot(epochs_list, accuracy_for_one, 'b', label='Accuracy for 1')

        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        self.trained_model = model

        return model #TODO - not correct thing to do?

    def predict(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            An array, this is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of belonging to the
            POSITIVE class (that had accidents)
        """

        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # YOUR CODE HERE
        #Convert to TensorDataset


        # Predict
        all_outputs = []
        all_labels = []
        all_raw = []

        # Convert from numpy to tensors for train data and corresponding labels
        # NB X_clean is already a numpy array
        #x_test = X_raw.to_numpy()
        #x_test.astype(float)

        x_test = torch.from_numpy(X_clean.astype(float))

        # Training dataset
        #train_ds = TensorDataset(x_train, y_train)
        #train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        #Also need to normalise data

        with torch.no_grad():
            outputs = self.trained_model(x_test.float())
            print(outputs)

        #Convert the outputs to probabilities
        sigmoid = nn.Sigmoid()
        predictions = sigmoid(outputs)
        print(predictions.flatten().numpy())

        return predictions.flatten().numpy() #PREDICTED CLASS LABELS (as probabilites)

        #     for data, labels in test_dl:
        #         outputs = self.trained_model(data.float())
        #
        #         # Merge outputs for auc roc score
        #         flat_raw = outputs.flatten().tolist()
        #         all_raw += flat_raw
        #
        #         # print(labels.float())
        #         # merge all labels
        #         flat_labels = labels.flatten().tolist()
        #         all_labels += flat_labels
        #
        #         # merge all outputs
        #         flat_outputs = F.sigmoid(outputs).round().flatten().tolist()
        #         all_outputs += flat_outputs
        #
        #         # all_outputs.append(F.sigmoid(outputs).round().flatten().tolist())
        #         # all_labels.append(labels.flatten().tolist())
        #
        # print(all_outputs)
        # print(all_labels)


    def evaluate_architecture(self, probabilities, labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """

        #Need to convert the probabilities into binary classification
        sigmoid = nn.Sigmoid()
        predicitions = sigmoid(torch.from_numpy(probabilities)).round()


        target_names = ['not claimed', 'claimed']

        print(classification_report(labels, predicitions, target_names=target_names))

        confusion_matrix(labels, predicitions)
        print(roc_auc_score(labels, probabilities))

        pass

    def save_model(self):
        # Please alter this file appropriately to work in tandem with your load_model function below
        with open('part2_claim_classifier.pickle', 'wb') as target:
            pickle.dump(self, target)


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    return trained_model

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """

    return  # Return the chosen hyper parameters

def main():
    print("testing")
    cc = ClaimClassifier()

    part2_training_data = "https://raw.githubusercontent.com/ben-xD/pricing-nn/master/part2_training_data.csv"
    df1 = pd.read_csv(part2_training_data)

    data = df1.drop(columns=["claim_amount", "made_claim"])
    labels = df1["made_claim"]

    #cc._preprocessor(data)

    cc.fit(data, labels)
    probabilities = cc.predict(data)
    cc.evaluate_architecture(probabilities, labels)

if __name__== "__main__":
    main()


