"""
@author: Hiromasa Kaneko
https://github.com/hkaneko1985/kennardstonealgorithm
"""
# select samples using Kennard-Stone algorithm
import numpy as np
import matplotlib.pyplot as plt

# --- input ---
# X : dataset of X-variables (samples x variables)
# k : number of samples to be selected
#
# --- output ---
# selectedsamplenumbers : selected sample numbers (training data)
# remainingsamplenumbers : remaining sample numbers (test data)

def kennardstonealgorithm( X, k ):
    X = np.array( X )
    originalX = X
    distancetoaverage = ( (X - np.tile(X.mean(axis=0), (X.shape[0], 1) ) )**2 ).sum(axis=1)
    maxdistancesamplenumber = np.where( distancetoaverage == np.max(distancetoaverage) )
    maxdistancesamplenumber = maxdistancesamplenumber[0][0]
    selectedsamplenumbers = list()
    selectedsamplenumbers.append(maxdistancesamplenumber)
    remainingsamplenumbers = np.arange( 0, X.shape[0], 1)
    X = np.delete( X, selectedsamplenumbers, 0)
    remainingsamplenumbers = np.delete( remainingsamplenumbers, selectedsamplenumbers, 0)
    for iteration in range(1, k):
        selectedsamples = originalX[selectedsamplenumbers,:]
        mindistancetoselectedsamples = list()
        for mindistancecalculationnumber in range( 0, X.shape[0]):
            distancetoselectedsamples = ( (selectedsamples - np.tile(X[mindistancecalculationnumber,:], (selectedsamples.shape[0], 1)) )**2 ).sum(axis=1)
            mindistancetoselectedsamples.append( np.min(distancetoselectedsamples) )
        maxdistancesamplenumber = np.where( mindistancetoselectedsamples == np.max(mindistancetoselectedsamples) )
        maxdistancesamplenumber = maxdistancesamplenumber[0][0]
        selectedsamplenumbers.append(remainingsamplenumbers[maxdistancesamplenumber])
        X = np.delete( X, maxdistancesamplenumber, 0)
        remainingsamplenumbers = np.delete( remainingsamplenumbers, maxdistancesamplenumber, 0)

    return(selectedsamplenumbers, remainingsamplenumbers)

def plot_ks(X,X_ks):
    plt.figure()
    plt.scatter( X[:,0], X[:,1], label = "all samples")
    plt.scatter( X[X_ks,0], X[X_ks,1], label = "selected samples")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc='upper right')
    plt.show()
    

# Same idea as above, but James made it look nice
def james_ksa(x: np.ndarray, k ):
    '''
    Selects k samples from x using the Kennard-Stone
    algorithm.

    Parameters
    ----------
    x:
        dataset

    k: int
        Number of samples to be selected

    Returns
        Tuple of two lists: (selectedsamplenumbers, remainingsamplenumbers)
    ----------

    '''
    # Make an original copy
    x_copy = x.copy()

    # Coerce into array
    x = np.array(x)

    distance_to_average = ( (x - np.tile(x.mean(axis=0), (x.shape[0], 1) ) )**2 ).sum(axis=1)

    #print(f'Distance_to_average: {distance_to_average}')
    # This gets a list of all entries where the distance to the average is equal to the max distance
    max_distance_sample_number = np.where( distance_to_average == np.max(distance_to_average) )
    #print(f'max_distance_sample_number: {max_distance_sample_number} ({len(max_distance_sample_number[0])})')

    # Take the first one?
    max_distance_sample_number = max_distance_sample_number[0][0]

    # Make a list that will contain the indices?
    selectedsamplenumbers = []

    # Add the maximum distance found
    selectedsamplenumbers.append(max_distance_sample_number)

    remainingsamplenumbers = np.arange(0, x.shape[0], 1)

    #print(f'remainingsamplenumbers: {remainingsamplenumbers} ({len(remainingsamplenumbers)})')
    x = np.delete(x, selectedsamplenumbers, 0)
    remainingsamplenumbers = np.delete( remainingsamplenumbers, selectedsamplenumbers, 0)
    #print(f'remainingsamplenumbers: {remainingsamplenumbers} ({len(remainingsamplenumbers)})')


    for iteration in range(1, k):

        selectedsamples = x_copy[selectedsamplenumbers]

        mindistancetoselectedsamples = []

        for mindistancecalculationnumber in range(0, x.shape[0]):

            distancetoselectedsamples = ((selectedsamples - np.tile(x[mindistancecalculationnumber], (selectedsamples.shape[0], 1)) )**2 ).sum(axis=1)
            mindistancetoselectedsamples.append(min(distancetoselectedsamples))

        maxdistancesamplenumber = np.where( mindistancetoselectedsamples == np.max(mindistancetoselectedsamples) )
        maxdistancesamplenumber = maxdistancesamplenumber[0][0]

        selectedsamplenumbers.append(remainingsamplenumbers[maxdistancesamplenumber])

        # This deletes the ones we already added
        x = np.delete(x, maxdistancesamplenumber, 0)
        remainingsamplenumbers = np.delete( remainingsamplenumbers, maxdistancesamplenumber, 0)

    return(selectedsamplenumbers, remainingsamplenumbers)