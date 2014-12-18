function demo_CV_classification_Issam
%% Cross Validation (Issam Laradji) demo
% This demo shows you how to use cross validation on a machine learning
% algorithm to select the best parameter value for 1 algorithm
% parameter.
%
% In this example we will use the ELM model and its 'nHidden' parameter - the
% number of hidden neurons to use in the hidden layer.
%
% We will show how cross-validation is used with and without earlyStop
% against 2000 training samples of the USPS dataset. We will also show the 
% effect of regularization on the ELM model against a synthetic dataset.
%
% Author: *Issam Laradji* (issam.laradji@gmail.com)

%% Test CV Model
clear all
close all

% Set random seed
rand('state', 1);

%% Load USPS dataset
load data/uspsData.mat

%% Choose a subset of the training set
X = X(1:2000,:);
y = y(1:2000,:);

fprintf('Running cross-validation with 2000 training samples of USPS\n');

% Create the ELM model and set the parameters for cross-validation
options.model = @matLearn_classification_ELM_Issam;
options.paramName = 'nHidden';
options.loss = 'zero one loss';
options.earlyStop = true;
options.paramValues = [5, 10, 50, 100, 500, 700, 800, 1000];

% Run cross-validation with 2 folds
options.nFolds = 2;

% Find the best model using multi-class cross-validation
[bestModel, bestParamValue, ~, ~] = matLearn_CV(X, y, options);

fprintf('Cross-validation done!\n');

% Compute the test error
yhat = bestModel.predict(bestModel, Xtest);
testError = sum(yhat~=ytest)/length(ytest);

fprintf('Best number of hidden neurons selected: %d\n', bestParamValue);
fprintf('Zero-one test error rate with ELM best model is: %.3f\n', testError);

% Plot 1 : Compare cross-validation with earlyStop=true and earlyStop=false
figure('position', [0, 100, 900, 700]);

% Set cross-validation folds to 2
options.nFolds = 2;

% Find the best model with earlyStop=true
options.earlyStop = true;
[~, ~, ~, validationErrorLog] =  matLearn_CV(X, y, options);

% Create sub-plot for earlyStop=true
subplot(2,2,1);
title('Cross-validation with earlyStop=true using the ELM model');
xlabel('Number of hidden neurons');
ylabel('Zero-one error rate (USPS validation set)');
hold on;
plot(validationErrorLog.paramValues, validationErrorLog.errorValues, ...
    'LineWidth',2);

% Find the best model with earlyStop=false
options.earlyStop = false;
[~, ~, ~, validationErrorLog] =  matLearn_CV(X, y, options);

% Create sub-plot for earlyStop=false
subplot(2,2,2);
title('Cross-validation with earlyStop=false using the ELM model');
xlabel('Number of hidden neurons');
ylabel('Zero-one error rate (USPS validation set)');
hold on;

plot(validationErrorLog.paramValues, validationErrorLog.errorValues, ...
    'LineWidth',2);

% Plot 2 : Compare ELM with high regularization strength
% against ELM with low regularization strength
clear all;

%% Load synthetic data
load data/data_exponential.mat 
X = Xtrain;
y = ytrain;
y(y==-1) = 2;

% Train ELM with low regularization strength
options = [];
options.lambda= 1;
elmModel = matLearn_classification_ELM_Issam(X, y, options);

% Create sub-plot for ELM decision boundary
subplot(2,2,3);
title('ELM with low regularization strength, lambda=1');
xlabel('Synthetic Dataset Feature 1');
ylabel('Synthetic Dataset Feature 2');
hold on;

plot2DClassifier(X, y, elmModel);

% Train ELM with high regularization strength
options.lambda= 1000;
elmModel = matLearn_classification_ELM_Issam(X, y, options);

% Create sub-plot for ELM decision boundary
subplot(2,2,4);
title(sprintf('ELM with high regularization strength, lambda=1000'));
xlabel('Synthetic Dataset Feature 1');
ylabel('Synthetic Dataset Feature 2');
hold on;

plot2DClassifier(X, y, elmModel);


function [] = plot2DClassifier(X, y, model)

increment = 500;

plot(X(y==1,1),X(y==1,2),'g+');hold on;
plot(X(y==2,1),X(y==2,2),'bo');

domain1 = xlim;
domain1 = domain1(1):(domain1(2)-domain1(1))/increment:domain1(2);
domain2 = ylim;
domain2 = domain2(1):(domain2(2)-domain2(1))/increment:domain2(2);

d1 = repmat(domain1',[1 length(domain1)]);
d2 = repmat(domain2,[length(domain2) 1]);

vals = model.predict(model,[d1(:) d2(:)]);


zData = reshape(vals,size(d1));
contourf(d1,d2,zData+rand(size(zData))/1000,[1 2 3],'k');
if all(zData(:) == 1)
    cm = [0 .5 0];
elseif all(zData(:) == 2)
    cm = [0 0 .5];
else
    cm = [0 0 .5;0 .5 0];
end
colormap(cm);

plot(X(y==1,1),X(y==1,2),'g+');hold on;
plot(X(y==2,1),X(y==2,2),'bo');
hold on;

