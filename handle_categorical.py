# import the packages
import numpy as np
import pandas as pd
import category_encoders as ce


def get_encoded_categorical(df):
    y = df[df.columns[0]] # to get first columne (target columne)

    # drop the first columne from dataframe and store it into new df_
    df_ = df.drop([df.columns[0]], axis = 1) 

    # get numerical variable
    num_cols = df._get_numeric_data().columns
    cols = df_.columns

    # get categorical variable 
    cate_cols = list(set(cols) - set(num_cols))

    # get binary encoded categorical variables
    encoder = ce.BinaryEncoder(cols=cate_cols)
    df_binary = encoder.fit_transform(df_)

    return df_binary

