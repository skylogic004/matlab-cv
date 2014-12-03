clear all
close all

%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
load data_regressOnOne.mat

% Do Regression CV by Matthew Dirks

fprintf('-----REGRESSION CV - NOT COMBNIED - SHOULD MATCH WITH COMBINED BELOW ----\n');
paramStruct = [];
paramStruct.name = 'fakeParam'; % Gives the name of the variable in 'options' to search over
paramStruct.values = [.0001 .001 .01 .1 0.4 0.5]; % Gives values that should be tried

optionsCV = [];
optionsCV.params = paramStruct;
% optionsCV.k = 2;
optionsCV.mode = 'leave-one-out';
optionsCV.shuffleData = false;
optionsCV.verbose = true;
optionsCV.p = 1;
[new_model] = matLearn_regression_CV(@matLearn_regression_regressOnOneMD, Xtrain, ytrain, optionsCV);

% Measure test error
yhat = new_model.predict(new_model,Xtest);
testError = mean(abs(yhat-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',new_model.name,testError);

%% Plot
plotRegression1D(Xtrain,ytrain,new_model);


fprintf('-----COMBINED----\n');
% Do combined CV 
optionsCV = [];
optionsCV.paramName = 'fakeParam';
optionsCV.paramValues = [.0001 .001 .01 .1 0.4 0.5];
% optionsCV.nFolds = 2;
optionsCV.leaveoneout = true;
optionsCV.shuffle = false;
optionsCV.model = @matLearn_regression_regressOnOneMD;
optionsCV.loss = 'absolute error';
[new_model, bestParamValue, bestError] = matLearn_CV(Xtrain, ytrain, optionsCV);

fprintf('Best param value %0.3f --> error = %0.3f\n', bestParamValue, bestError);

% Measure test error
yhat = new_model.predict(new_model,Xtest);
testError = mean(abs(yhat-ytest));
fprintf('Averaged absolute test error with %s is: %.3f\n',new_model.name,testError);

%% Plot
plotRegression1D(Xtrain,ytrain,new_model);