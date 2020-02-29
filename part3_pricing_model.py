import pandas as pd
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

from statistics import mean
import matplotlib.pyplot as plt


from NeuralNet import NeuralNet


def fit_and_calibrate_classifier(classifier, X, y):
    # DO NOT ALTER THIS FUNCTION
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, train_size=0.85, random_state=0)
    classifier = classifier.fit(X_train, y_train)

    # This line does the calibration for you
    calibrated_classifier = CalibratedClassifierCV(
        classifier, method='sigmoid', cv='prefit').fit(X_cal, y_cal)
    return calibrated_classifier


# class for part 3
class PricingModel():
    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY
    def __init__(self, calibrate_probabilities=False):
        """
        Feel free to alter this as you wish, adding instance variables as
        necessary.
        """
        self.y_mean = None
        self.calibrate = calibrate_probabilities

        self.scaler = None
        self.trained_model = None
        # =============================================================
        # READ ONLY IF WANTING TO CALIBRATE
        # Place your base classifier here
        # NOTE: The base estimator must have:
        #    1. A .fit method that takes two arguments, X, y
        #    2. Either a .predict_proba method or a decision
        #       function method that returns classification scores
        #
        # Note that almost every classifier you can find has both.
        # If the one you wish to use does not then speak to one of the TAs
        #
        # If you wish to use the classifier in part 2, you will need
        # to implement a predict_proba for it before use
        # =============================================================
        self.base_classifier = None # ADD YOUR BASE CLASSIFIER HERE


    # YOU ARE ALLOWED TO ADD MORE ARGUMENTS AS NECESSARY TO THE _preprocessor METHOD
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
        X: ndarray
            A clean data set that is used for training and prediction.
        """
        # =============================================================
        # YOUR CODE HERE
        lb = preprocessing.LabelBinarizer()
        print("X_raw", type(X_raw))
        features = X_raw.drop(columns=['id_policy', 'pol_bonus', 'pol_sit_duration', 'pol_insee_code'], axis=1)
        temp = pd.get_dummies(features.pol_coverage)
        features = features.drop(columns=['pol_coverage'], axis=1)
        features = pd.concat([features, temp], axis=1)
        temp = pd.get_dummies(features.pol_pay_freq)
        features = features.drop(columns=['pol_pay_freq'], axis=1)
        features = pd.concat([features, temp], axis=1)
        features.pol_payd = lb.fit_transform(features.pol_payd)
        temp = pd.get_dummies(features.pol_usage)
        features = features.drop(columns=['pol_usage'], axis=1)
        features = pd.concat([features, temp], axis=1)
        features.drv_drv2 = lb.fit_transform(features.drv_drv2)
        features = features.drop(columns=["drv_age2"])
        features.drv_sex1 = lb.fit_transform(features.drv_sex1)
        features = features.drop(columns=["drv_sex2"])
        features = features.drop(columns=["drv_age_lic1", "drv_age_lic2"])
        temp = pd.get_dummies(features.vh_fuel)
        features = pd.concat([features, temp], axis=1)
        features = features.drop(columns=['vh_fuel'])
        features = features.drop(columns=['vh_model', 'vh_make'])
        features.vh_type = lb.fit_transform(features.vh_type)
        features = features.drop(columns=["town_mean_altitude", "town_surface_area","population", "commune_code", "canton_code", "city_district_code", "regional_department_code"])
        normalised_feature_array = preprocessing.MinMaxScaler().fit_transform(features)
        
        return normalised_feature_array


    def fit(self, X_raw, y_raw, claims_raw, weighting=9, learning_rate=0.001, batch_size=20, num_epochs=10, hidden_size=50):
        """Classifier training function.

        Here you will use the fit function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded
        y_raw : ndarray
            A one dimensional array, this is the binary target variable
        claims_raw: ndarray
            A one dimensional array which records the severity of claims

        Returns
        -------
        self: (optional)
            an instance of the fitted model
        """
        nnz = np.where(claims_raw != 0)[0]
        self.y_mean = np.mean(claims_raw[nnz]) #Average of all claims
        # =============================================================


        # THE FOLLOWING GETS CALLED IF YOU WISH TO CALIBRATE YOUR PROBABILITES
        # if self.calibrate:
        #     self.base_classifier = fit_and_calibrate_classifier(
        #         self.base_classifier, X_clean, y_raw)
        # else:
        #     self.base_classifier = self.base_classifier.fit(X_clean, y_raw)
        # return self.base_classifier

        # Shuffle data
        state = np.random.get_state()
        X_raw = X_raw.sample(frac=1).reset_index(drop=True)
        np.random.set_state(state)
        y_raw = y_raw.sample(frac=1).reset_index(drop=True)

        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # Split data
        percentile_60 = int(X_clean.shape[0] * 0.6)
        percentile_80 = int(X_clean.shape[0] * 0.8)

        train_data = X_clean[:percentile_60]
        train_labels = y_raw[:percentile_60]

        test_data = X_clean[percentile_60:percentile_80]
        test_labels = y_raw[percentile_60:percentile_80]
        self.test_data = test_data
        self.test_labels = test_labels

        val_data = X_clean[percentile_80:]
        val_labels = y_raw[percentile_80:]

        # Convert from numpy to tensors for train data and corresponding labels
        # NB X_clean is already a numpy array
        x_train = torch.tensor(train_data)
        y_train = torch.tensor(train_labels)

        x_test = torch.tensor(test_data)
        y_test = torch.tensor(test_labels.values)

        x_val = torch.tensor(val_data)
        y_val = torch.tensor(val_labels.values)

        # Training dataset
        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Testing dataset
        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Validations dataset
        val_ds = TensorDataset(x_val, y_val)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Input and Output
        num_inputs = train_data.shape[1]
        output_size = 1

        # Create a model with hyperparameters
        model = NeuralNet(num_inputs, hidden_size, output_size)

        # Weight positive samples higher
        pos_weight = torch.ones([1])
        pos_weight.fill_(weighting)

        # Loss criterion and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        epochs_list = []
        training_loss = []
        batch_loss = []

        for epoch in range(num_epochs):
            for xb, yb in train_dl:
                # Forwards pass
                preds = model(xb.float())  # Why do I need to add float() here?
                loss = criterion(preds.flatten(), yb.float())

                # TODO - delete this: For calculating the average loss and accuracy
                batch_loss.append(loss.item())
                #print(loss)
                # batch_accuracy.append(model.accuracy(preds, yb, batch_size))

                # Backward and optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epochs_list.append(epoch)
            print(epoch)
            training_loss.append(mean(batch_loss))
            # accuracy_list.append(mean(batch_accuracy))

        #plt.plot(epochs_list, accuracy_for_one, 'b', label='Accuracy for 1')

        #plt.plot(epochs_list, training_loss, 'g', label='Training loss')
        #plt.title('Training loss')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.show()

        self.trained_model = model

    def predict_claim_probability(self, X_raw):
        """Classifier probability prediction function.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : ndarray
            This is the raw data as downloaded

        Returns
        -------
        ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO A SIMILAR LINE TO THE FOLLOWING SOMEWHERE IN THE CODE
        X_clean = self._preprocessor(X_raw)

        # Predict
        #all_outputs = []
        #all_labels = []
        #all_raw = []

        x_test = torch.tensor(X_clean)

        with torch.no_grad():
            outputs = self.trained_model(x_test.float())
            # Convert the outputs to probabilities
            predictions = F.sigmoid(outputs)
            print("prediction shape: ", predictions.shape)
            # print(predictions.flatten().numpy())

        return predictions.flatten().numpy()  # return probabilities for the positive class (label 1)


    def predict_premium(self, X_raw):
        """Predicts premiums based on the pricing model.

        Here you will implement the predict function for your classifier.

        Parameters
        ----------
        X_raw : numpy.ndarray
            A numpy array, this is the raw data as downloaded

        Returns
        -------
        numpy.ndarray
            A one dimensional array of the same length as the input with
            values corresponding to the probability of beloning to the
            POSITIVE class (that had accidents)
        """
        # =============================================================
        # REMEMBER TO INCLUDE ANY PRICING STRATEGY HERE.
        # For example you could scale all your prices down by a factor

        return self.predict_claim_probability(X_raw) * self.y_mean

    #TODO - delete
    def evaluate_architecture(self, probabilities, labels):
        """Architecture evaluation utility.

        Populate this function with evaluation utilities for your
        neural network.

        You can use external libraries such as scikit-learn for this
        if necessary.
        """
        sigmoid = nn.Sigmoid()
        predictions = probabilities.round()
        #print(predictions)

        target_names = ['not claimed', 'claimed']
        print(f'labels {labels}')
        print(f'predictions {predictions}')
        print(f'probabilities {probabilities}')

        print(classification_report(labels.astype(int), predictions.astype(int)))

        print(confusion_matrix(labels, predictions))
        auc_score = roc_auc_score(labels, probabilities)
        print(f'auc: {auc_score}')
        print("Accuracy: ", accuracy_score(labels, predictions))
        #print("Labels: ", labels)
        #print("Predictions: ", predictions)
        return auc_score

    def save_model(self):
        """Saves the class instance as a pickle file."""
        # =============================================================
        # File name was changed, according to Marek's Piazza
        with open('part3_pricing_model_linear.pickle', 'wb') as target:
            pickle.dump(self, target)

    def get_test_data(self):
        return [self.test_data, self.test_labels]


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part3_pricing_model_linear.pickle', 'rb') as target:
        return pickle.load(target)

def main():
    # ClaimClassifierHyperParameterSearch()

    #Load pandas dataframe from csv
    df1 = pd.read_csv('part3_training_data.csv')
    y_raw = df1["made_claim"]
    claims_raw = df1["claim_amount"]
    features = df1.drop(columns=["made_claim", "claim_amount"])
    print(type(features))

    pricingModel = PricingModel()
    pricingModel.fit(features, y_raw, claims_raw)
    pricingModel.save_model()

    [test_data, test_labels] = pricingModel.get_test_data()

    probabilities = pricingModel.predict_claim_probability(test_data)

    #predictions = pricingModel.predict_premium(test_data)

    pricingModel.evaluate_architecture(probabilities, test_labels.to_numpy())

if __name__ == "__main__":
    main()
