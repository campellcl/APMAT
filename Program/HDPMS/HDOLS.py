"""
HDOLS
    Hiker Distance Ordinary Least Squares (HDOLS) script predicts a validated hiker's distance in miles-per-day.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse


def main(input_data_dir, model_storage_dir):
    model_train_data_fname = 'df_train.csv'
    model_test_data_fname = 'df_test.csv'
    df = pd.read_csv(input_data_dir + "\DistancePrediction.csv")
    '''Perform noise removal on the dataframe'''
    # Remove any training and test data with MPD less than 0 or > 50:
    df = df[df['MPD'] > 0]
    df = df[df['MPD'] <= 50]
    # Remove any entries with a location and no associated direction:
    df = df.loc[~df['LOCDIR'].str.endswith("UD")]
    # Perform dtype conversions so the model does not treat categorical data as numeric:
    df['LOCDIR'] = df['LOCDIR'].astype('category')
    df['HID'] = df['HID'].astype('category')
    # Update the representation of the dataframe in memory:
    df = df.copy()
    # Partition the dataframe into train test splits:
    df_train, df_test = train_test_split(df, test_size=0.4, random_state=0)
    # Check to see if the model exists already:
    if model_train_data_fname not in os.listdir(input_data_dir)\
            or model_test_data_fname not in os.listdir(input_data_dir):
        print("Model stored in HikerData\DistancePrediction.csv not found! Generating new model...")
        # Create OLS Regression model post outlier removal:
        ols_model = smf.ols(formula='MPD ~ LOCDIR + HID + 1', data=df_train).fit()
        print(ols_model.params)
        print(ols_model.summary())
        y_pred = ols_model.predict(df_test)
        # y_pred = np.exp(y_pred)
        # Add a column to the test data frame for the MPD predicted by the model:
        df_test = df_test.copy()
        df_test['MPD_PRED'] = y_pred
        df_test.to_csv(input_data_dir + "\df_test.csv")
        # df_train = df_train.copy()
        # df_train['MPD_PRED'] = np.exp(ols_model.predict(df_train))
        df_test.to_csv(input_data_dir + "\df_train.csv")
        # Evaluate model accuracy using Root Mean Squared Error (RMSE):
        root_mean_squared_error = rmse(y_pred, df_test['MPD'])
        print("Model RMSE (Miles-Per-Day): %d" % root_mean_squared_error)
    else:
        df_test = pd.read_csv(input_data_dir + "\df_test.csv", index_col=0)
        df_train = pd.read_csv(input_data_dir + "\df_train.csv", index_col=0)
        y_pred = df_test['MPD_PRED'].values
        root_mean_squared_error = rmse(y_pred, df_test['MPD'])
        print("Model RMSE (Miles-Per-Day): %f" % root_mean_squared_error)
        x_bin_edges = np.arange(0, 50, 1)
        y_bin_edges = np.arange(0, 25, 1)
        plt.hist2d(x=df_test.MPD, y=df_test.MPD_PRED, bins=[x_bin_edges, y_bin_edges], hold=True)
        plt.title("Testing Data Predicted vs. Actual")
        plt.xlabel("Hiker Miles-Per-Day Actual")
        plt.ylabel("Hiker Miles-Per-Day Predicted")
        plt.colorbar()
        ols_line_best_fit = smf.ols(formula='MPD_PRED ~ MPD + 1', data=df_test).fit()
        pred_regression_y_max = ols_line_best_fit.predict({'MPD': max(df_test.MPD)})
        pred_regression_y_min = ols_line_best_fit.predict({'MPD': min(df_test.MPD)})
        print("pred_reg_y_max: %d" % pred_regression_y_max)
        print("pred_reg_y_min: %d" % pred_regression_y_min)
        plt.plot([min(df_test.MPD), max(df_test.MPD)], [pred_regression_y_min, pred_regression_y_max], c='red', linewidth=2)
        plt.show()


if __name__ == '__main__':
    hiker_journal_entry_csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../..', 'Data/HikerData/'
    ))
    storage_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../..', 'Data/HikerData/DistancePrediction.csv'
    ))
    main(input_data_dir=hiker_journal_entry_csv_path, model_storage_dir=storage_dir)
