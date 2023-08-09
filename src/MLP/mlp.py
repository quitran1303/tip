
#Model: MLP Model

# Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import argparse
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os.path

# Main Function

parser = argparse.ArgumentParser(description='Traffic Forecasting with MLPRegressor')
parser.add_argument('-f')  

parser.add_argument('--dataset', type=str, default='EBSP')
parser.add_argument('--pred_len', type=int, default=15)
parser.add_argument('--times', type=int, default=1)
args = parser.parse_args()

def save_files(y_true, y_pred, direction, pred_len):
    save_path = '.'
    date = datetime.now().strftime('%Y%m%d')
    true_file = f'y_true_{date}.{direction}.{pred_len}_mins.csv'
    pred_file = f'y_pred_{date}.{direction}.{pred_len}_mins.csv'
    np.savetxt(os.path.join(save_path,true_file), y_true, delimiter=",")
    np.savetxt(os.path.join(save_path,pred_file), y_pred, delimiter=",")
    print("SAVE - True file : ",os.path.join(save_path,true_file), ", prediction file: ", os.path.join(save_path,pred_file))
    return True


def run_model(dataset, pred_len, sensorid):
    print("sensor Id:", sensorid)
    # Hyper parameters
    test_size = 0.2
    validation_size = 0.1
    layers = (32, 8)
    Lambda = 1e-05
    batch_size = 64
    initial_learning_rate = 0.01
    epochs = 500
    
    readFile = ''
    # Reading the timesere dataset
    if dataset == "EBSP":
        readFile = 'ebspeed.csv'
    
    if dataset == "WBSP":
        readFile = 'wbspeed.csv'
        
    df = pd.read_csv(readFile)[sensorid][1:-1]
    df = df.dropna()
    df = np.array([df[i:i + pred_len] for i in range(len(df) - pred_len - 1 )])

    # Getting the input/output
    X = df[:, :-1]
    Y = df[:, -1:]

    # Split data into Train/Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=1)

    # Normalizing the data
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    scaler_y = StandardScaler().fit(Y_train)
    Y_train_sc = scaler_y.transform(Y_train)
    Y_test_sc = scaler_y.transform(Y_test)
    
    # Initializing the regressor model
    regressor = MLPRegressor(
			hidden_layer_sizes=layers,
			activation='logistic',
			solver='adam',
			alpha=Lambda,
			batch_size=batch_size,
			learning_rate='adaptive',
			learning_rate_init=initial_learning_rate,
			max_iter=epochs,
			shuffle=True,
			tol=0.0001,
			verbose=False,
			early_stopping=True,
			validation_fraction=validation_size)
            
    # Train the regressor
    avg_score = 0
    avg_error = 0
    min_error = 1
    max_error = 0
    avg_conv_iter = 0
    for i in range(args.times) :
        print("iteration: ", i+1)
        regressor.fit(X_train_sc, Y_train_sc.ravel()) 
        avg_score += regressor.best_validation_score_
        avg_conv_iter += regressor.n_iter_
        cur_error = 1 - regressor.best_validation_score_
        avg_error += cur_error
        min_error = min(cur_error, min_error)
        max_error = max(cur_error, max_error)
        print("Done iteration", i+1)
        
    avg_score /= args.times
    avg_error /= args.times
    avg_conv_iter /= args.times
    
    print('avg_score    :', round(avg_score, 2))
    print('avg_error    :', round(avg_error, 2))
    print('max_error    :', round(max_error, 2))
    print('min_error    :', round(min_error, 2))
    print('avg_conv_iter:', round(avg_conv_iter, 2))
    print('Regressor Score:', regressor.score(X_test_sc, Y_test_sc))
    Y_pred_sc = regressor.predict(X_test_sc)
    Y_pred_real = scaler_y.inverse_transform(Y_pred_sc)
    Y_test1 = pd.DataFrame(Y_test)
    Y_pred_real1 = pd.DataFrame(Y_pred_real)

    #save files
    save_files(Y_test1, Y_pred_real1, dataset+'.'+sensorid, pred_len)



if __name__ == '__main__':
    print('Traffic Forecasting with MLPRegressor')
    print("dataset", args.dataset)
    print("pred_len:", args.pred_len)
    print("times:", args.times)
    print("-------------")
    
    sensorList = []
    if args.dataset == "EBSP":
        sensorList = ["WE1","WE2","WE3","WE4","WE5","WE6","WE7","WE8","WE9","WE10","WE11","WE12","WE13","WE14","WE15","WE16","WE17"] #EBSP
    if args.dataset == "WBSP":
        sensorList = ["EW1","EW2","EW3","EW4","EW5","EW6","EW7","EW8","EW9","EW10","EW11","EW12","EW13","EW14","EW15","EW16","EW17"] #WBSP

    for sId in sensorList:
        run_model(args.dataset, args.pred_len, sId)
    
    
    print("DONE")