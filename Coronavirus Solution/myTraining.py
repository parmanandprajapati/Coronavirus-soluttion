from threading import main_thread
import pandas as pd
import numpy as np
from pip import main
from sklearn.linear_model import LogisticRegression
import pickle





def data_split(data,ratio):
    np.random.seed(42) # it freeze the data
    shuffled= np.random.permutation(len(data)) 
    test_set_size= int(len(data)*ratio)  
    test_indices = shuffled[:test_set_size] 
    train_indices= shuffled[test_set_size:]  

    return data.iloc[train_indices],data.iloc[test_indices]



if __name__ == "__main__":

    #Read the data
    df= pd.read_csv('data.csv')
    train,test= data_split(df, 0.2)
    X_train = train[['fever','bodyPain',"age","runnyNose",'diffBreath']].to_numpy() 
    X_test = test[['fever','bodyPain',"age","runnyNose",'diffBreath']].to_numpy()

    Y_train = train[['infectionProb']].to_numpy().reshape(2060 ,)
    Y_test = test[['infectionProb']].to_numpy().reshape(515 ,) 

    clf= LogisticRegression()
    clf.fit(X_train,Y_train)
     
    # using picke for storing and dump

    file=open('model.pkl','wb')
    pickle.dump(clf,file)
    
    file.close()
    
    #code for inference
