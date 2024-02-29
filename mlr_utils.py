# This is a page of basic utility functions that are unlikely to be changed by the average user.
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, train_test_split


def remove_prefix_zeros(number: str) -> int:
    """
    This function removes the zeros from the front of a number read in as a string, then returns the number as an int. 
    It allows you to read in the data from a spreadsheet with less problems.
    This should ultimately be redundant once we figure out input formating.
    """

    temp = number
    loop = True
    while(loop):
        if temp.startswith('0'):
            temp=temp[1:]
        else:
            loop=False
            
    try:
        temp=float(temp)
        temp=int(temp)
        temp=str(temp)
    except ValueError:
        temp=str(temp)
    
    return int(temp)


def r2_val(y_test,y_pred_test,y_train):
    """
    Calculates the external R2 pred as described:
    https://pdfs.semanticscholar.org/4eb2/5ff5a87f2fd6789c5b9954eddddfd1c59dab.pdf
    """

    y_resid = y_pred_test - y_test
    SS_resid = np.sum(y_resid**2)
    y_var = y_test - np.mean(y_train)
    SS_total = np.sum(y_var**2)
    r2_validation = 1-SS_resid/SS_total
    return(r2_validation)


def repeated_k_fold(X_train, y_train,reg = LinearRegression(), k:int = 3, n:int = 100):
    """
    Reapeated k-fold cross-validation. 
    For each of n repeats, the (training)data is split into k folds. 
    For each fold, this part of the data is predicted using the rest. 
    Once this is done for all k folds, the coefficient of determination (R^2) of the predictions of all folds combined (= the complete data set) is evaluated
    This is repeated n times and all n R^2 are returned for averaging/further analysis
    """
    
    rkf = RepeatedKFold(n_splits=k, n_repeats=n)
    r2_scores = []
    y_validations = np.zeros((np.shape(X_train)[0],n))
    y_predictions = np.zeros((np.shape(X_train)[0],n))
    foldcount = 0

    for i,foldsplit in enumerate(rkf.split(X_train)):
        fold = i%k # Which of k folds 
        rep = int(i/k) # Which of n repeats
        model = reg.fit(X_train[foldsplit[0]],y_train[foldsplit[0]]) # Train on the subset of data identified by foldsplit[0]
        y_validations[foldcount:foldcount+len(foldsplit[1]), rep] = y_train[foldsplit[1]] # foldsplit[1]: validation fold
        y_predictions[foldcount:foldcount+len(foldsplit[1]), rep] = model.predict(X_train[foldsplit[1]])
        foldcount += len(foldsplit[1])
        if fold+1==k:
            foldcount = 0
    r2_scores = np.asarray([metrics.r2_score(y_validations[:,rep], y_predictions[:,rep]) for rep in range(n)])
    return(r2_scores)

def train_test_splits(temp_data_df:pd.DataFrame, split:str, test_ratio:float, randomstate:int = 0, defined_training_set:list[int] = [], defined_test_set:list[int] = [], subset:list[int] = [], verbose:bool = True) -> tuple[list[int], list[int]]:
    """
    Given the main dataframe and some parameters, return lists of y index values for a training and test set.
    For historical y_equidist, use y_equidist_old.  Be warned though, it drops points with the same y value as the min or max y value.

    :temp_data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :split: 'random', 'ks', 'y_equidist', 'define', 'none'; Type of split to use
    :test_ratio: Ratio of the data to use as a test set
    :randomstate: Seed to use when chosing the random split
    :defined_training_set: Y indexes corresponding to a manual training set. Only used if split == 'define'
    :defined_test_set: Y indexes corresponding to a manual test set. Only used if split == 'define'
    :subset: The subset of y indexes to use for another split method, originally used for MLR after a classification algorithm
    :verbose: Whether to print the extended report
    """

    import kennardstonealgorithm as ks

    
    if (subset == []):
        data_df = temp_data_df.copy()
    else:
        data_df = temp_data_df.loc[subset, :].copy()

    x = data_df.iloc[:, 2:].to_numpy() # Array of just feature values (X_sel)
    y = data_df.iloc[:, 0].to_numpy() # Array of response values (y_sel)
    test_size = int(len(data_df.index)*test_ratio) # Number of points in the test set
    train_size = len(data_df.index) - test_size

    if split == "random":
        random.seed(a = randomstate)
        test_set = random.choices(list(data_df.index), k = test_size)
        training_set = [x for x in data_df.index if x not in test_set]

    elif split == "ks":
        # There may be some issues with test_set_index being formatted as an array and training_set_index being a list
        training_set_index, test_set_index = ks.kennardstonealgorithm(x, train_size)
        training_set = list(data_df.index[training_set_index])
        test_set = list(data_df.index[test_set_index])


    elif split == "y_equidist_old":
        no_extrapolation = True
        # Only difference I can see between extrapolation and no_extrapolation is that no_e cuts off the highest and lowest y values first
        
        if no_extrapolation:
            # This block seems to chop the min and max values off of the y_array
            minmax = [np.argmin(y),np.argmax(y)]
            y_ks = np.array(([i for i in y if i not in [np.min(y),np.max(y)]]))  # THIS IS THE PROBLEM LINE.  IT DROPS ALL POINTS WITH OUTPUT EQUAL TO THE MIN OR MAX VALUE.
            y_ks_indices = [i for i in range(len(y)) if i not in minmax]

            print(f'length of y array: {len(y)}')
            print(f'shape of y_ks: {np.shape(y_ks)}')
            print(f'length of y_ks_indices: {len(y_ks_indices)}')

            # indices relative to y_ks:
            y_ks_formatted = y_ks.reshape(np.shape(y_ks)[0], 1)
            print(f'Shape of y_ks_formatted: {np.shape(y_ks_formatted)}')
            VS_ks,TS_ks = ks.kennardstonealgorithm(y_ks_formatted, test_size)
            # indices relative to y_sel:
            TS_ = sorted([y_ks_indices[i] for i in list(TS_ks)]+minmax) # HERE IT ADDS minmax BACK IN, BUT NOT THE OTHER POINTS THAT WERE REMOVED.
            VS_ = sorted([y_ks_indices[i] for i in VS_ks])

        else:
            VS_,TS_ = ks.kennardstonealgorithm(y.reshape(np.shape(y)[0],1),int((test_ratio)*np.shape(y)[0]))

        training_set = list(data_df.index[TS_])
        test_set = list(data_df.index[VS_])

    elif split == "y_equidist":
        no_extrapolation = True
        # Only difference I can see between extrapolation and no_extrapolation is that no_e cuts off the highest and lowest y values first
        
        if no_extrapolation:
            # This block seems to chop the min and max values off of the y_array
            # minmax = [np.argmin(y),np.argmax(y)]
            # y_ks = np.array(([i for i in y if i not in [np.min(y),np.max(y)]]))  # THIS IS THE PROBLEM LINE.  IT DROPS ALL POINTS WITH OUTPUT EQUAL TO THE MIN OR MAX VALUE.
            # y_ks_indices = [i for i in range(len(y)) if i not in minmax]

            # Rewritten from above to keep track of which points were removed for being equal to the min or max value
            y_min = np.min(y)
            y_max = np.max(y)
            y_ks = np.array(([i for i in y if i not in [y_min,y_max]]))
            y_ks_indices = [i for i, val in enumerate(y) if val != y_min and val != y_max]
            y_not_ks_indices = [i for i, val in enumerate(y) if val == y_min or val == y_max]

            # indices relative to y_ks:
            y_ks_formatted = y_ks.reshape(np.shape(y_ks)[0], 1)
            VS_ks,TS_ks = ks.kennardstonealgorithm(y_ks_formatted, test_size)

            # indices relative to y_sel:
            TS_ = sorted([y_ks_indices[i] for i in list(TS_ks)]+y_not_ks_indices) # Replaced minmax with y_not_ks_indices
            VS_ = sorted([y_ks_indices[i] for i in VS_ks])

        else:
            VS_,TS_ = ks.kennardstonealgorithm(y.reshape(np.shape(y)[0],1),int((test_ratio)*np.shape(y)[0]))

        training_set = list(data_df.index[TS_])
        test_set = list(data_df.index[VS_])

    elif split == 'define':
        training_set = defined_training_set
        test_set = defined_test_set

    elif split == "none":
        training_set = data_df.index.to_list()
        test_set = []

    else: 
        raise ValueError("split option not recognized")
    
    if(verbose):
        y_train = data_df.loc[training_set, 'response']
        y_test = data_df.loc[test_set, 'response']

        if (len(training_set) + len(test_set) == len(data_df.index)):
            print('All indices accounted for!')
        else:
            print('Missing indices!')

        print("Training Set size: {}".format(len(training_set)))
        print("Training Set mean: {:.3f}".format(np.mean(y_train)))
        print("Test Set size: {}".format(len(test_set)))
        print("Test Set mean: {:.3f}".format(np.mean(y_test)))
        # print("Shape X_train: {}".format(X_train.shape))
        # print("Shape X_test:  {}".format(X_test.shape))   
        plt.figure(figsize=(5, 5))
        hist, bins = np.histogram(y,bins="auto")#"auto"
        plt.hist(y_train, bins, alpha=0.5, label='y_train',color="black")
        plt.hist(y_test, bins, alpha=0.5, label='y_test')
        # plt.legend(loc='best')
        plt.xlabel("Output",fontsize=20)
        plt.ylabel("N samples",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()

    return training_set, test_set