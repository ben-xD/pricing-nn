import numpy as np
import pickle
import pandas as pd
from NeuralNet import NeuralNet
from statistics import mean
import random                                                                                                                                                                                                                
from copy import deepcopy

#PYTORCH
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset

# SKLEARN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import preprocessing


class ClaimClassifier():                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                              
    def __init__(self):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        self.trained_model = None                                                                                                                                                                                                                                             
        self.scaler = None
        self.initial_params = {'learning_rate': 0.01, 'batch_size': 10, 'num_epochs': 30, 'layer_size': 30}  
                                                                                                                                                                                                                                                                    
    def _preprocessor(self, X_raw):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        if self.scaler == None:                                                                                                                                                                                                                                               
            self.scaler = preprocessing.MinMaxScaler()                                                                                                                                                                                                                        
        # Normalisation (also saved for testing data later)                                                                                                                                                                                                                   
        return self.scaler.fit_transform(X_raw)                                                                                                                                                                                                                               
                                                                                                                                                                                                                                                                              
    def fit(self, X_raw, y_raw, hparams=None, weighting=1, num_hidden_layers=2, layer_sizes=None):                                                                                                                                                    
                                                                                                                                                                                                                                                              
        # Shuffle data                                                                                                                                                                                                                                                        
        state = np.random.get_state()                                                                                                                                                                                                                                         
        X_raw = X_raw.sample(frac=1).reset_index(drop=True)                                                                                                                                                                                                                   
        np.random.set_state(state)                                                                                                                                                                                                                                            
        y_raw = y_raw.sample(frac=1).reset_index(drop=True)                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
        #Unpack hyperparameters
        if(hparams is None):
          hparams = self.initial_params

        learning_rate = hparams["learning_rate"]                                                                                                                                                                                                                              
        batch_size = hparams["batch_size"]                                                                                                                                                                                                                                    
        num_epochs = hparams["num_epochs"]                                                                                                                                                                                                                                    
        layer_size = hparams["layer_size"]

        if(layer_sizes is None):
          layer_sizes = [layer_size*num_hidden_layers]
                                                                                                                                                                                                                                                                              
        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE                                                                                                                                                                                                           
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
        self.val_data = val_data                                                                                                                                                                                                                                            
        self.val_labels = val_labels                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                              
        # Convert from numpy to tensors for train data and corresponding labels                                                                                                                                                                                                                                                                                                                                                                                                                              
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

        #Create the model                                                                                                                                                                                                                                                                      
        model = NeuralNet(num_inputs, num_hidden_layers, layer_sizes, output_size)                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                              
        # Weighting for positve class                                                                                                                                                                                                                                      
        pos_weight = torch.ones([1])                                                                                                                                                                                                                                          
        pos_weight.fill_(weighting)                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                              
        # Loss criterion and optimizer                                                                                                                                                                                                                                        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)                                                                                                                                                                                                               
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        for epoch in range(num_epochs):                                                                                                                                                                                                                                       
            for xb, yb in train_dl:                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                              
                # Forwards pass                                                                                                                                                                                                                                               
                preds = model(xb.float())  # Why do I need to add float() here?                                                                                                                                                                                               
                loss = criterion(preds.flatten(), yb.float())                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                # Backward and optimize                                                                                                                                                                                                                                       
                loss.backward()                                                                                                                                                                                                                                               
                optimizer.step()                                                                                                                                                                                                                                              
                optimizer.zero_grad()                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                              
        self.trained_model = model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                              
    def predict(self, X_raw):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                              
        # REMEMBER TO HAVE THE FOLLOWING LINE SOMEWHERE IN THE CODE                                                                                                                                                                                                           
        X_clean = self._preprocessor(X_raw)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        x_test = torch.tensor(X_clean)                                                                                                                                                                                                                                        

        # Convert the outputs to probabilities                                                                                                                                                                                                                                                                      
        with torch.no_grad():                                                                                                                                                                                                                                                 
            outputs = self.trained_model(x_test.float())                                                                                                                                                                                                                                                                                                                                                                                                                                           
            predictions = F.sigmoid(outputs)

        # Return predicted class labels (as probabilites)                                                                                                                                                                                                                                                                      
        return predictions.flatten().numpy()                                                                                                                                                                                        
                                                                                                                                                                                                                                                                              
    def evaluate_architecture(self, probabilities, labels):                                                                                                                                                                                                                   
        """Architecture evaluation utility.                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                              
        Populate this function with evaluation utilities for your                                                                                                                                                                                                             
        neural network.                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                              
        You can use external libraries such as scikit-learn for this                                                                                                                                                                                                          
        if necessary.                                                                                                                                                                                                                                                         
        """                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                              
        # Convert probabilities into binary classification                                                                                                                                                                                                        
        sigmoid = nn.Sigmoid()                                                                                                                                                                                                                                                
        predictions = probabilities.round()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                              
        target_names = ['not claimed', 'claimed']                                                                                                                                                                                                                             

        #Metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        print(confusion_matrix(labels, predictions))

        precision = precision_score(labels, predictions)                                                                                                                                                                                                                                
        print("Precision: ", precision)

        recall = recall_score(labels, predictions)                                                                                                                                                                                                                                
        print("Recall: ", recall)

        auc_score = roc_auc_score(labels, probabilities)                                                                                                                                                                                                                      
        print("AUC ROC: ", auc_score)

        accuracy = accuracy_score(labels, predictions)                                                                                                                                                                                                                                
        print("Accuracy: ", accuracy)                                                                                                                                                                                                            
                                                                                                                                                                                                                              
        return auc_score, accuracy                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                              
    def save_model(self):                                                                                                                                                                                                                                                     
        # Please alter this file appropriately to work in tandem with your load_model function below                                                                                                                                                                          
        with open('part2_claim_classifier.pickle', 'wb') as target:                                                                                                                                                                                                           
            pickle.dump(self, target)

    def get_initial_params(self):
        return self.initial_params

    def get_test_data(self):                                                                                                                                                                                                                                                  
        return [self.test_data, self.test_labels]

    def get_val_data(self):                                                                                                                                                                                                                                                  
        return [self.val_data, self.val_labels]


def load_model():
    # Please alter this section so that it works in tandem with the save_model method of your class
    with open('part2_claim_classifier.pickle', 'rb') as target:
        return pickle.load(target)

# ENSURE TO ADD IN WHATEVER INPUTS YOU DEEM NECESSARRY TO THIS FUNCTION
def ClaimClassifierHyperParameterSearch():
    """Performs a hyper-parameter for fine-tuning the classifier.

    Implement a function that performs a hyper-parameter search for your
    architecture as implemented in the ClaimClassifier class. 

    The function should return your optimised hyper-parameters. 
    """
    df1 = pd.read_csv('part2_training_data.csv')
    X = df1.drop(columns=["claim_amount", "made_claim"])
    y = df1["made_claim"]

    claimClassifier = ClaimClassifier()
    hparams = claimClassifier.get_initial_params()

    num_hidden_layers, layer_sizes = LayerHyperParameterSearch(X, y, hparams)

    trial_params = {                                                                                                                                                                                                                                                          
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],                                                                                                                                                                                                                                 
        'batch_size': [1, 10, 20, 40, 80, 160, 320, 640],                                                                                                                                                                                                                                      
        'num_epochs': [5,10,15,20, 25, 30, 35, 40, 45, 50, 75, 100],                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    }                                                                                                                                                                                                                                                                         

    num_cycles = 1
    trials = 0

    all_hparams_tried = {}
    best_auc_score = 0
    best_hparams = None

    for cycles in range(num_cycles):                                                                                                                                                                                                                                                                
      for tkey in trial_params.keys():                                                                                                                                                                                                                                          
          print(tkey, trial_params[tkey])
          auc_scores = []  
          for tparam in trial_params[tkey]:
                                                                                                                                                                                                                                          
            trials += 1

            #Set hyperparameter                                                                                                                                                                                                                                           
            hparams[tkey] = tparam                                                                                                                                                                                                                                        
            print(f'Using: {hparams}')                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                          
            #train                                                                                                                                                                                                                                                        
            claimClassifier.fit(X, y, hparams=hparams, num_hidden_layers=num_hidden_layers, layer_sizes=layer_sizes)                                                                                                                                                                                                                    
            [val_data, val_labels] = claimClassifier.get_val_data()                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                          
            #predict                                                                                                                                                                                                                                                      
            probabilities = claimClassifier.predict(pd.DataFrame(val_data)) 

            auc_score, accuracy = claimClassifier.evaluate_architecture(probabilities, val_labels)                                                                                                                                                                                 
            auc_scores.append(auc_score)

            #Save the score and hyperparameter for every hparam combination tried
            all_hparams_tried[auc_score] = hparams

            if(auc_score > best_auc_score):
              best_auc_score = deepcopy(auc_score)
              best_hparams = hparams

          #Pick the best auc score and set it before the next parameter is trialled
          auc_scores_np = np.asarray(auc_scores)
          print(auc_scores)
          print(auc_scores_np)
          best_index = np.argmax(auc_scores_np)  
          best_hparam_value = trial_params[tkey][best_index]

          #Set these hparams at the starting point for the next cycle
          hparams[tkey] = best_hparam_value
          best_auc_score = auc_scores_np[best_index]
              
    print(f'Best hyperparameters: {best_hparams} with auc_score: {best_auc_score}')

    #Evaluate the tuned model on the test set and save this best model
    [test_data, test_labels] = claimClassifier.get_test_data()
    claimClassifier.fit(X, y, hparams=hparams, num_hidden_layers=num_hidden_layers, layer_sizes=layer_sizes)                                                                                                                                                                                                                    
    claimClassifier.save_model()

    probabilities = claimClassifier.predict(pd.DataFrame(test_data))
    final_auc, final_accuracy = claimClassifier.evaluate_architecture(probabilities, test_labels)
    print(f'AUC ROC: {final_auc} and Accuracy: {final_accuracy}')

    return hparams

def LayerHyperParameterSearch(X, y, hparams):

    claimClassifier = ClaimClassifier()                                                                                                                                                                                                                                       

    #Number of layers to test
    MAX_LAYER = 33
    num_layers = [1,2,4,8,16,32, MAX_LAYER]
    layer_sizes = list(range(1, MAX_LAYER+1))

    auc_scores = []
    accuracy_scores = []
    best_num_hidden_layers = None
    best_layer_sizes = None
    best_auc_score = 0
    
    #Create models with different numbers of hidden layers
    for num_hidden_layers in num_layers:

      random.shuffle(layer_sizes)

      #Create model, test and evaluate for each
      claimClassifier.fit(X, y, hparams=hparams, num_hidden_layers=num_hidden_layers, layer_sizes=layer_sizes)
      [val_data, val_labels] = claimClassifier.get_val_data()
      probabilities = claimClassifier.predict(pd.DataFrame(val_data))
      model_auc, model_accuracy = claimClassifier.evaluate_architecture(probabilities, val_labels)

      auc_scores.append(model_auc)
      accuracy_scores.append(model_accuracy)

      if(model_auc > best_auc_score):
        best_auc_score = deepcopy(model_auc)
        best_num_hidden_layers = deepcopy(num_hidden_layers)
        best_layer_sizes = deepcopy(layer_sizes)

    all_multiplyers = [0.5, 1, 2, 3, 4, 5]

    #Create models with varying sizes of neurons in each of those layers
    for multiplyer in all_multiplyers:
      multiplied_layer_sizes = [int(i*multiplyer) for i in layer_sizes]
      print('muliplyed: ', multiplied_layer_sizes)

      #Create model, test and evaluate for each
      claimClassifier.fit(X, y, hparams=hparams, num_hidden_layers=best_num_hidden_layers, layer_sizes=multiplied_layer_sizes)
      [val_data, val_labels] = claimClassifier.get_val_data()
      probabilities = claimClassifier.predict(pd.DataFrame(val_data))
      model_auc, model_accuracy = claimClassifier.evaluate_architecture(probabilities, val_labels)

      if(model_auc > best_auc_score):
        best_auc_score = deepcopy(model_auc)
        best_layer_sizes = deepcopy(multiplied_layer_sizes)   
    
    return best_num_hidden_layers, best_layer_sizes

# We found the optimal hyperparameters using ClaimClassifierHyperParameterSearch
# This is to avoid re-running the code again.
def test_save_and_load():
    df1 = pd.read_csv('part2_training_data.csv')
    X = df1.drop(columns=["claim_amount", "made_claim"])
    y = df1["made_claim"]
  
    # train here
    claimClassifier = ClaimClassifier()
    weighting = 9
    claimClassifier.fit(X, y, weighting=weighting)
    claimClassifier.save_model()

    # tested here
    claimClassifier = load_model()
    [test_data, test_labels] = claimClassifier.get_test_data()
    probabilities = claimClassifier.predict(pd.DataFrame(test_data))
    claimClassifier.evaluate_architecture(probabilities, test_labels)

def main():
    ClaimClassifierHyperParameterSearch()
    # test_save_and_load()

if __name__ == "__main__":
    main()