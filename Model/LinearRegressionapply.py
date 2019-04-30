import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, Perceptron
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor


def store_csv(complete_data,df):
    my_df1 = pd.DataFrame(complete_data)
    my_df1 = my_df1.T
    df = df.append(my_df1)
    return df

def Model_predict(df):
    essay_Set = [1,2,3,4,5,6,7,8,9,10]
    test_essay = pd.read_csv("/mnt/1f2870f0-1578-4534-b33f-0817be64aade/projects/automaticscriiptEval/incedo_participant/test_dataset.csv")
    for i in essay_Set:
        print(i)
        train_file = pd.read_csv("/mnt/1f2870f0-1578-4534-b33f-0817be64aade/projects/automaticscriiptEval/TrainData/datatrain{}.csv".format(i))
        test_file = pd.read_csv("/mnt/1f2870f0-1578-4534-b33f-0817be64aade/projects/automaticscriiptEval/TestData/datatest{}.csv".format(i))
        set_filter = (test_essay.Essayset == float(i))
        id_list = list(test_essay[set_filter]['ID'])            #List containing ID
        set_list = list(test_essay[set_filter]['Essayset'])     #List containing set number
        list_data = []
        final = []
        X_train = train_file.drop(['score'],1)
        y_train = train_file['score']
        model = OneVsRestClassifier(GradientBoostingRegressor())
        model.fit(np.nan_to_num(X_train),np.nan_to_num(y_train))
        X_test = test_file
        labels = model.predict(np.nan_to_num(X_test))
        for j in labels:
            list_data.append(round(j))
        final.append(id_list)
        final.append(set_list)
        final.append(list_data)
        df = store_csv(final,df)
        df.to_csv("submission.csv",index=False, header=False)

if __name__ == "__main__":
    df = pd.DataFrame([['id','essay_set','essay_score']])
    Model_predict(df)
