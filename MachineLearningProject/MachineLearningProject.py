#/*
# * The MIT License
# *
# * Copyright 2021 Ethan Welborn - ethan.welborn@go.tarleton.edu.
# *
# * Permission is hereby granted, free of charge, to any person obtaining a copy
# * of this software and associated documentation files (the "Software"), to deal
# * in the Software without restriction, including without limitation the rights
# * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# * copies of the Software, and to permit persons to whom the Software is
# * furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in
# * all copies or substantial portions of the Software.
# *
# * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# * THE SOFTWARE.
# */

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.utils import resample
from matplotlib import pyplot as plt

class australianWeatherModel():
    REMOVE_CORRELATED_VARIABLES = None;
    REMOVE_CORRELATED_VARIABLES_THRESHOLD= None;
    REMOVE_DATE= None;

    CLASSIFICATION_MODEL= None; # alwaysDry, naive, randomForest

    RANDOM_FOREST_ESTIMATORS= None;

    RESAMPLE= None; # downSample, upSample, any other value for no resampling

    def getModelParameters(self):
        return {
            "REMOVE_CORRELATED_VARIABLES": self.REMOVE_CORRELATED_VARIABLES,
            "REMOVE_CORRELATED_VARIABLES_THRESHOLD": self.REMOVE_CORRELATED_VARIABLES_THRESHOLD,
            "REMOVE_DATE": self.REMOVE_DATE,
            "CLASSIFICATION_MODEL": self.CLASSIFICATION_MODEL,
            "RANDOM_FOREST_ESTIMATORS": self.RANDOM_FOREST_ESTIMATORS,
            "RESAMPLE": self.RESAMPLE,
        };

    def __init__(self,REMOVE_CORRELATED_VARIABLES=True,REMOVE_CORRELATED_VARIABLES_THRESHOLD=0.8,REMOVE_DATE=True,CLASSIFICATION_MODEL="randomForest",RANDOM_FOREST_ESTIMATORS=100,RESAMPLE="downSample"):
        self.REMOVE_CORRELATED_VARIABLES = REMOVE_CORRELATED_VARIABLES;
        self.REMOVE_CORRELATED_VARIABLES_THRESHOLD = REMOVE_CORRELATED_VARIABLES_THRESHOLD;
        self.REMOVE_DATE = REMOVE_DATE;
        self.CLASSIFICATION_MODEL = CLASSIFICATION_MODEL;
        self.RANDOM_FOREST_ESTIMATORS = RANDOM_FOREST_ESTIMATORS;
        self.RESAMPLE = RESAMPLE;

    # Returns the confusion matrix of the model
    def model(self,printModelParameters=False,printModelProgress=False):
        if printModelParameters:
            print("REMOVE_CORRELATED_VARIABLES =",self.REMOVE_CORRELATED_VARIABLES);
            print("REMOVE_CORRELATED_VARIABLES_THRESHOLD =",self.REMOVE_CORRELATED_VARIABLES_THRESHOLD);
            print("REMOVE_DATE =",self.REMOVE_DATE);
            print("CLASSIFICATION_MODEL =",self.CLASSIFICATION_MODEL);
            print("RANDOM_FOREST_ESTIMATORS =",self.RANDOM_FOREST_ESTIMATORS);
            print("RESAMPLE =",self.RESAMPLE);
            print("");

        rawDataFrame = pd.read_csv("weatherAUS.csv");

        # Remove all rows with missing data
        rawDataFrame = rawDataFrame.dropna();

        # Get a copy of the raw dataframe without our dependent variable
        dataFrame = rawDataFrame.drop(["RainTomorrow"],axis=1);

        # Remove "Date" from the data, as it's not particularly useful to our classification (Could lead to overfitting,
        #   and it's difficult to scale to a number value)
        if self.REMOVE_DATE == True:
            dataFrame = dataFrame.drop(["Date"],axis=1);

        # Remove all of the discrete/categorical columns from our data
        #dataFrame = dataFrame.drop(["Location"],axis=1);
        #dataFrame = dataFrame.drop(["WindGustDir"],axis=1);
        #dataFrame = dataFrame.drop(["WindDir9am"],axis=1);
        #dataFrame = dataFrame.drop(["WindDir3pm"],axis=1);
        #dataFrame = dataFrame.drop(["RainToday"],axis=1);

        def mapColumnToDictionary(column,dict):
            i = 0;
            for location in dataFrame[column].unique():
                dict[location] = i;
                i = i + 1;
            dataFrame[column] = dataFrame[column].map(dict);

        locationToIndexDictionary = {};
        windGustDirToIndexDictionary = {};
        windDir9amToIndexDictionary = {};
        windDir3pmToIndexDictionary = {};
        rainTodayToIndexDictionary = {};

        if self.REMOVE_DATE == False:
            mapColumnToDictionary("Date",locationToIndexDictionary);
        mapColumnToDictionary("Location",locationToIndexDictionary);
        mapColumnToDictionary("WindGustDir",windGustDirToIndexDictionary);
        mapColumnToDictionary("WindDir9am",windDir9amToIndexDictionary);
        mapColumnToDictionary("WindDir3pm",windDir3pmToIndexDictionary);
        mapColumnToDictionary("RainToday",rainTodayToIndexDictionary);

        # Determine which columns are heavily correlated and remove them
        # https://stackoverflow.com/questions/49282049/remove-strongly-correlated-columns-from-dataframe
        def trimm_correlated(df_in, threshold):
            df_corr = df_in.corr(method='pearson', min_periods=1)
            df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
            un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
            df_out = df_in[un_corr_idx]
            return df_out

        originalColumns = dataFrame.columns;
        if self.REMOVE_CORRELATED_VARIABLES:
            dataFrame = trimm_correlated(dataFrame,self.REMOVE_CORRELATED_VARIABLES_THRESHOLD);
        newColumns = dataFrame.columns;

        #print(dataFrame);
        if printModelProgress:
            for column in originalColumns:
                found = False;
                for nColumn in newColumns:
                    if column == nColumn:
                        found = True;
                if not found:
                    print(column,"was removed");

        X_train,X_test,y_train,y_test = train_test_split(dataFrame,rawDataFrame["RainTomorrow"],test_size=0.3);

        # https://elitedatascience.com/imbalanced-classes Sections 1 and 2
        if self.RESAMPLE == "downSample" or self.RESAMPLE == "upSample":
            X_train = X_train.assign(RainTomorrow = y_train);
            majority = X_train[X_train.RainTomorrow == "No"];
            minority = X_train[X_train.RainTomorrow == "Yes"];

            if self.RESAMPLE == "downSample":
                majorityDownSampled = resample(majority,replace=False,n_samples=len(minority));
                X_train = pd.concat([majorityDownSampled,minority]);
        
            elif self.RESAMPLE == "upSample":
                minorityUpSampled = resample(minority,replace=True,n_samples=len(majority));
                X_train = pd.concat([majority,minorityUpSampled]);

            y_train = X_train["RainTomorrow"];
            X_train = X_train.drop("RainTomorrow",axis=1);

        classificationModel = None;
        if self.CLASSIFICATION_MODEL == "alwaysDry":
            class alwaysDryClassifier():
                def fit(X_train, y_train):
                    pass;
                def predict(X_test):
                    return ["No" for _ in range(len(X_test))];

            classificationModel = alwaysDryClassifier;
        elif self.CLASSIFICATION_MODEL == "naive":
            classificationModel = GaussianNB();
        elif self.CLASSIFICATION_MODEL == "randomForest":
            classificationModel = RandomForestClassifier(n_estimators=self.RANDOM_FOREST_ESTIMATORS);

        if printModelProgress:
            print("Training model...");
        classificationModel.fit(X_train, y_train);
        if printModelProgress:
            print("Training complete!");

        y_pred = classificationModel.predict(X_test);

        return confusion_matrix(y_test, y_pred);
        #print(classification_report(y_test, y_pred));

data = [];

def addModelToData(givenModel,title):
    cm = givenModel.model(printModelParameters=True,printModelProgress=True)
    print(cm);
    
    correctDryDayPredictions = cm[0][0];
    correctRainyDayPredictions = cm[1][1];
    totalDryDayPredictions = cm[0][0]+cm[1][0];
    totalRainyDayPredictions = cm[1][1]+cm[0][1];

    if totalDryDayPredictions == 0:
        dryDayAccuracy = 0;
    else:
        dryDayAccuracy = math.floor((correctDryDayPredictions/totalDryDayPredictions)*1000)/10;
    
    if totalRainyDayPredictions == 0:
        rainyDayAccuracy = 0;
    else:
        rainyDayAccuracy = math.floor((correctRainyDayPredictions/totalRainyDayPredictions)*1000)/10;

    generalAccuracy = math.floor(((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0]))*1000)/10;
    data.append((
        title,
        dryDayAccuracy,
        rainyDayAccuracy,
        generalAccuracy
    ));
    print("Accuracy for correctly predicting a dry day:",dryDayAccuracy,"%");
    print("Accuracy for correctly predicting a rainy day:",rainyDayAccuracy,"%");
    print("Accuracy for correct predictions in general:",generalAccuracy,"%");
    print("");
    print("--------------");
    print("");

alwaysDryModel = australianWeatherModel(CLASSIFICATION_MODEL="alwaysDry");
addModelToData(alwaysDryModel,"alwaysDry");

for classificationModel in ("naive","randomForest"):
    for _resample in (None,"downSample","upSample"):
        #if classificationModel == "randomForest":
        #    for _forestSize in (100,300):
        #        model = australianWeatherModel(CLASSIFICATION_MODEL=classificationModel,RESAMPLE=_resample,RANDOM_FOREST_ESTIMATORS=_forestSize);
        #        addModelToData(model,classificationModel + ", " + (_resample or "no resample") + ", forestSize " + str(_forestSize));
        #else:
        #    model = australianWeatherModel(CLASSIFICATION_MODEL=classificationModel,RESAMPLE=_resample);
        #    addModelToData(model,classificationModel + ", " + (_resample or "no resample"));
        model = australianWeatherModel(CLASSIFICATION_MODEL=classificationModel,RESAMPLE=_resample);
        addModelToData(model,classificationModel + ", " + (_resample or "no resample"));

#cm = australianWeatherModel().model();
#print(cm);
#corrects = np.trace(cm);
#total = np.sum(cm);
#print(f"The number of corrects is : {corrects} while the accuracy is: {corrects/total*100}");

for dataPoint in data:
    print(dataPoint);

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

fig,ax = plt.subplots();
x = np.arange(len(data));
width = 0.25;
dryDayBar = ax.bar(x-width, list(dp[1] for dp in data), width, label="Dry Day Prediction Accuracy");
rainyDayBar = ax.bar(x, list(dp[2] for dp in data), width, label="Rainy Day Prediction Accuracy");
generalDayBar = ax.bar(x+width, list(dp[3] for dp in data), width, label="General Prediction Accuracy");

ax.bar_label(dryDayBar, padding=3);
ax.bar_label(rainyDayBar, padding=3);
ax.bar_label(generalDayBar, padding=3);

ax.set_xticks(x);
ax.set_xticklabels(list(dp[0] for dp in data));

ax.legend();

#plt.bar(list(dp[0]-0.25 for dp in data), list(dp[1] for dp in data), color="r", width=0.25);
#plt.bar(list(dp[0] for dp in data), list(dp[2] for dp in data), color="g", width=0.25);
#plt.bar(list(dp[0]+0.25 for dp in data), list(dp[3] for dp in data), color="b", width=0.25);

plt.show()