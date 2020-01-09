# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:38:58 2019
@author: achauhan39
"""

import pickle
from ac_util import tune_it, eval_model, get_sepsis_train_test_data, RANDOM_STATE
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#from vecstack import stacking

OUTPUT_DIR = "../../output/"    

BEST = 1

# Gradient Boosting
def test_gb(X,Y):
    clf = GradientBoostingClassifier(random_state= RANDOM_STATE)
    params = { 'max_depth' : list(range(1,20,2)) ,
               'n_estimators' : [10,30,50,70],
               'min_samples_split' : [2,4,6,10],
               'learning_rate' : [.01, .1],
               'loss' : ['deviance', 'exponential'] }
    if(BEST) : 
         params = {"max_depth" : [3], "min_samples_split"  : [2], "n_estimators" : [100],  #180
          "learning_rate": [.1] ,"loss" : ['deviance'] }          
    best_clf = tune_it(clf, X,Y, params, "GRAD BOOST")         
    
    return best_clf   


# logistic Regression
def test_lr(X, Y) :
    clf = LogisticRegression(solver = 'lbfgs', random_state = RANDOM_STATE)
    params = { 'max_iter' : [100, 200],
               'C' : [0.5, 1.0 , 1.5 ]
             }
    if(BEST) :
        params = { 'max_iter' : [200], 'C' : [1.0] }
    best_clf = tune_it(clf, X, Y, params, "LR")
    
    return best_clf



# SVC
def test_svc(X,Y):
    clf = SVC(probability = True, gamma = 'scale', random_state = RANDOM_STATE)
    C_lin = [0.1]
    C_rbf = [0.01, 0.1, 1.0, 2.0]
    params= [{ 'C' : C_lin, 'kernel': 'linear' } ,
             #{ 'C' : C_rbf, 'kernel' : ['poly'] , degree :[3,5,7,9] }
            ]
    if(BEST) :
        params = { 'C' : [1.0], 'kernel' : ['poly'] }
    
    best_clf = tune_it(clf, X, Y , params)
    
    return best_clf



# MLP
def test_mlp(X,Y):
    clf = MLPClassifier(early_stopping=True, random_state=RANDOM_STATE)
    params = {'activation': ['relu' ,'logistic'],
              'alpha': [0.001, 0.01, 0.1],
              'hidden_layer_sizes':[(100,),(500,), (1000,), (500,60)],
              'solver' : ['lbfgs' ,'adam'],   
              'max_iter' : [2000] ,
              'learning_rate': ['constant','adaptive'] }         
    if(BEST) : 
        params = {'activation': ['relu'],
              'alpha': [0.001],
              'hidden_layer_sizes':[(100,50)],
              'solver' : ['adam'],   
              'max_iter' : [2000] ,
              'learning_rate': ['adaptive'] } 
        
    best_clf = tune_it(clf,X,Y, params, "MLP") 
    return best_clf

def save_model(clf, name):
    with open (OUTPUT_DIR + name + '.pkl' , 'wb') as outfile:
        pickle.dump(clf, outfile)
    outfile.close()
        
    
def load_model(name):
    with open (OUTPUT_DIR + name + '.pkl' , 'rb') as infile:
        clf = pickle.load(infile)
    infile.close()
    return clf


    
#def run_GB_x() :
#    obsv_win = [12, 10, 8, 6 ]
#    for t in obsv_win:
#        print("------- OBSV_WINDOW = {} --------------".format(t) )
#        X_train, X_test, Y_train , Y_test = get_sepsis_train_test_data(pred_w = 6 , obsv_w = t )
#        best_clf  = test_gb(X_train, Y_train)
#        eval_model(best_clf, X_train, X_test, Y_train ,Y_test,  "GB")
        

def run_GB() :
    
    time_windows = [(6,12) , (8,10) ,(10,8), (12,6) ]
    
    for pred, obsv in time_windows:
        print("------PRED_WINDOW = {} , OBSV_WINDOW = {} --------------".format(pred , obsv) )
        X_train, X_test, Y_train , Y_test = get_sepsis_train_test_data(pred_w = pred , obsv_w = obsv )
        best_clf  = test_gb(X_train, Y_train)
        eval_model(best_clf, X_train, X_test, Y_train ,Y_test,  "GB")
        

def get_best_models():
    
    X_train, X_test, Y_train , Y_test = get_sepsis_train_test_data(pred_w = 6 , obsv_w = 12)
    
    # 1. Gradient Boosting
    best_clf  = test_gb(X_train, Y_train)    
    eval_model(best_clf, X_train, X_test, Y_train ,Y_test,  "GradientBoost")
    save_model(best_clf, "gradientBoost")
    
    # 2. SVM
    best_clf  = test_svc(X_train, Y_train)
    eval_model(best_clf, X_train, X_test, Y_train ,Y_test,  "SVC")
    save_model(best_clf, "svm")
    
    #3. Logistic Regression
    best_clf  = test_lr(X_train, Y_train)
    eval_model(best_clf, X_train, X_test, Y_train ,Y_test,  "Log-Regression")
    save_model(best_clf, "logisticRegression")
    
    # 4. MLP
    best_clf  = test_mlp(X_train, Y_train)
    eval_model(best_clf, X_train, X_test, Y_train ,Y_test,  "MLP")
    save_model(best_clf, "mlp")
    
    # Run Gradient-boost model for different observation windows (12 to 6 hours)
    # run_GB()


def run_pickled_models():
    X_train, X_test, Y_train , Y_test = get_sepsis_train_test_data()
    
    models = ['gradientBoost', 'svm' , 'logisticRegression' , 'mlp' ]
    for m in models :
        clf = load_model(m)
        eval_model(clf, X_train, X_test, Y_train ,Y_test, m)
    
    
if __name__ == '__main__' :
    get_best_models()    
   # run_pickled_models()
    
