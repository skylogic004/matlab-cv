clear all
close all

%% Load synthetic {Xtrain,ytrain} and {Xtest,ytest}
load data_exponential.mat

%% Broken Stump model

% Train broken stump model
options = [];
[model_stump] = matLearn_classification2_brokenStump(Xtrain,ytrain,options);

% Test broken stump model
yhat = model_stump.predict(model_stump,Xtest);

% Measure test error
testError = sum(yhat~=ytest)/length(ytest);
fprintf('Averaged absolute test error with %s is: %.3f\n',model_stump.name,testError);

% Plot model predictions
plot2DClassifier(Xtrain,ytrain,model_stump);

%% Exponential-Loss model

% Train exponential loss model
options.addBias = 1;
options.lambdaL2 = 1;
[model_exp] = matLearn_classification2_exponential(Xtrain,ytrain,options);

% Test exponential loss model
yhat = model_exp.predict(model_exp,Xtest);

% Measure test error
testError = sum(yhat~=ytest)/length(ytest);
fprintf('Averaged absolute test error with %s is: %.3f\n',model_exp.name,testError);

% Plot model predictions
plot2DClassifier(Xtrain,ytrain,model_exp);