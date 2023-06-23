##############################################################
# This code has been used to generate the results taht we reported
# in the conference paper with the following details:
# Authors: Oguz Toragay, Shaheen Pouya, Mehrdad Mohammadi
# Title: How Do Machine Learning Models Perform in
# Predicting the Solution Time for Optimization
# Problems? Case of Job Shop Scheduling Problem.
# Please cite our paper if you use the provided code 
##############################################################

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
import keras_tuner
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperParameters

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Data loader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
loader = "C:\\Users\szp0155\OneDrive - Auburn University\Desktop\Gap-Time\\Neural\data_2.csv"
df = pd.read_csv(loader)

# df = df.drop(['Machine'], axis=1)
# df = df.drop(['Job'], axis=1)
# df = df.drop(['Gap 5'], axis=1)
# df = df.drop(['Gap 10'], axis=1)

print(df.head())

instances = list(df.columns)
instances.remove('Gap 0')
X = df[instances].values
y = df[['Gap 0']].values

Pr_Sc =StandardScaler()
Target_Sc=StandardScaler()
Pr_ScN =MinMaxScaler()
Target_ScN=MinMaxScaler()

Xs= Pr_Sc.fit_transform(X)
ys= Target_Sc.fit_transform(y)
Xn= Pr_ScN.fit_transform(X)
yn= Target_ScN.fit_transform(y)

rnd=random.randint(1,100)
mape_mean , r2_mean =[],[]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def linear(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, (y.ravel()), test_size=0.2, random_state=rnd)

    ####### Used Linear Regression methods
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import BayesianRidge
    from sklearn.ensemble import GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    from sklearn.tree import DecisionTreeRegressor
    #model = DecisionTreeRegressor(max_depth=12)
    from sklearn.preprocessing import PolynomialFeatures
    #model = make_pipeline(PolynomialFeatures(degree=3, include_bias=True),LinearRegression() )


    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return y_test, y_pred

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SVR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def svr (X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, (y.ravel()), test_size=0.2, random_state=rnd)

    model = make_pipeline(StandardScaler(), SVR(kernel = 'rbf', C=1000, gamma = 0.0001, epsilon = 0.01))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return y_test, y_pred

## Cross Validation
def svr_cv(X, y):

    model = make_pipeline(StandardScaler(), SVR(kernel = 'rbf', C=1000, gamma = 0.0001, epsilon = 0.01))

    scoring = make_scorer(mean_absolute_percentage_error)
    #scoring =make_scorer(r2_score)

    scores = cross_validate(model, X, y.ravel(), cv=25, scoring=scoring, return_train_score=True)
    print(np.mean(scores['test_score']))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tuning SVR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def svr_tuner(X, y):
    parameters = [{'kernel': ['rbf','linear'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2],'C': [10, 100, 1000, 10000]}]
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_absolute_percentage_error)
    svr = GridSearchCV(SVR(epsilon = 0.01), parameters, cv = 5, scoring=scorer)
    svr.fit(X, y.ravel())

    print("Grid scores on training set:")
    means = svr.cv_results_['mean_test_score']
    stds = svr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  MLP Regression  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mlp(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, (y.ravel()), test_size=0.2, random_state=rnd)
    model = MLPRegressor(hidden_layer_sizes=(50, 100, 50), max_iter=2000, activation='relu', solver='adam',
                          learning_rate='constant', alpha=0.0001)



    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return y_test, y_pred

##  Cross Validation
def mlp_cv(X, y):

    model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), max_iter=2000, activation='logistic', solver='adam',
                         learning_rate='adaptive', alpha=0.0001)
    scoring = make_scorer(mean_absolute_percentage_error)
    #scoring =make_scorer(r2_score)

    scores = cross_validate(model, X, y.ravel(), cv=10, scoring=scoring)#, return_train_score=True)
    print('CV_Score:', np.mean(scores['test_score']))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MLP Tuner ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mlp_tuner(X, y):
    estimator=MLPRegressor()
    scoring = make_scorer(mean_absolute_percentage_error)
    param_grid = {'hidden_layer_sizes': [(50, 50), (100, 100), (50,50,50), (50,100,50)],
              'activation': ['relu','tanh','logistic'],
              'alpha': [1e-4, 1e-3, 0.01],
              'learning_rate': ['constant','adaptive'], 'max_iter':[2000],
              'solver': ['adam']}
    gsc = GridSearchCV(
        estimator,
        param_grid,
        cv=15, scoring=scoring, verbose=3, n_jobs=-1)
    gsc.fit(X, y.ravel())

    best_params = gsc.best_params_
    print('Best Parameters:', best_params)
    # best_mlp = MLPRegressor(hidden_layer_sizes = best_params["hidden_layer_sizes"],
    #                         activation =best_params["activation"],
    #                         solver=best_params["solver"],
    #                         max_iter= 5000, n_iter_no_change = 200
    #                         )
    #scoring = make_scorer(mean_absolute_percentage_error)
    #scores = cross_validate(best_mlp, X, y, cv=10, scoring=scoring, return_train_score=True, return_estimator = True)

    return best_params

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tuning Deep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def deep_tuner (X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, (y.ravel()), test_size=0.2, random_state=rnd)

    def build_model(hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 10)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                   activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4,5e-2, 5e-3, 5e-4 ,2e-3, 3e-3, 1e-4 ])),
            loss=tf.keras.losses.MeanAbsolutePercentageError(),
            )
        return model

    build_model(HyperParameters())
    tuner = RandomSearch(build_model,objective=keras_tuner.Objective("val_loss", direction="min"),max_trials=500)

    tuner.search(X_train, y_train, epochs=1500, validation_data=(X_test, y_test))
    best_model = tuner.get_best_models()[0]

    print(tuner.results_summary())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Deep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def deep(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, (y.ravel()), test_size=0.2, random_state=rnd)


    model = Sequential()
    model.add(Dense(224, input_shape=(len(instances),), activation='relu'))       #0
    model.add(Dense(256, activation='relu'))      #1
    model.add(Dense(96, activation='relu'))       #2
    model.add(Dense(256, activation='relu'))      #3
    model.add(Dense(32, activation='relu'))       #4
    model.add(Dense(32, activation='relu'))       #5

    model.add(Dense(1, activation='linear'))

    model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                loss=tf.keras.losses.MeanAbsolutePercentageError(),
            )
    model.fit(X_train, y_train,  epochs=1500)

    y_pred = model.predict(X_test)
    return y_test, y_pred

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Testing & Accuracy ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def scorer(y_test, y_pred):
    tester = pd.DataFrame(y_test, columns =['test'])
    predict = pd.DataFrame(y_pred, columns = ['predict'])

    combined = pd.concat([tester, predict], axis = 1)
    saver(combined)
    mape = round(mean_absolute_percentage_error(y_test, y_pred),3)
    r2 = round(r2_score(y_test, y_pred), 3) * 100

    print(combined)
    print('MAPE:  ', mape)
    print('R Square:', r2)

    return mape, r2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   Saver ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def saver(combined):
    dft = pd.DataFrame.from_dict(combined)
    dft.to_csv('Results_Vs_Prediction.csv', mode='a', index=True)
    print('file saved!!')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Use Here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


y_test, y_pred = deep(X,y)
scorer(y_test, y_pred)


quit()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Average Generator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for i in range(25):
    y_test, y_pred = deep(Xs,ys)

    print('\nRound', i)
    rnd = random.randint(1, 100)
    mape, r2 = scorer(y_test, y_pred)
    mape_mean.append(mape)
    r2_mean.append(r2)

print('\n\n',df.head())
print('\n\nAverage MAPE:  ' , round(np.mean(mape_mean),3))
print('Average R Square:' , round(np.mean(r2_mean),1))
print(mape_mean)
print(r2_mean)

