import pandas as pd

from Model.LinearRegressionapply import Model_predict
from test_extractor import Datatest
from train_extractor import DataTrain

if __name__ == "__main__":
    df = pd.DataFrame([['id','essay_set','essay_score']])
    print("Training data Preprocessing Loading...")
    DataTrain()
    print("Testing data Preprocessing Loading...")
    Datatest()
    print("Model training Loading...")
    Model_predict(df)
    print("check file submission.csv in Model Directory")
