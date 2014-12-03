clear all
close all

%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
load data_regressOnOne.mat

optionsCV = [];
optionsCV.paramName = 'fakeParam';
optionsCV.paramValues = {'a', 'b', 'winning-value', 'c'};

% optionsCV.nFolds = 2;
optionsCV.leaveoneout = true;
optionsCV.shuffle = false;
optionsCV.model = @matLearn_regression_CV_testModel_stringParams;
optionsCV.loss = 'absolute error';
[new_model, bestParamValue, bestError] = matLearn_CV(Xtrain, ytrain, optionsCV);

fprintf('Best param value %s --> error = %0.3f\n', bestParamValue, bestError);

% Measure test error
yhat = new_model.predict(new_model,Xtest);
testError = mean(abs(yhat-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',new_model.name,testError);

%% Plot
plotRegression1D(Xtrain,ytrain,new_model);