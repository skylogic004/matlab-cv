function [model] = matLearn_regression_regressOnOne(X,y,options)
% matLearn_regression_regressOnOne(X,y,options)
%
% Description:
%   - Minimizes the squared error for one feature
%
% Options:
%   - selectedFeature: The feature to use for the univariate regression
%
% Authors:
% 	- Mark Schmidt (2014)

[nTrain,nFeaturs] = size(X);

[selectedFeature, fakeParam] = myProcessOptions(options,'selectedFeature',1, 'fakeParam', 'blah');

% fprintf('sel feat: %d, fake param: %0.1f\n', selectedFeature, fakeParam);

% Compute the regression weight w for feature j, ignoring the others
j = 1;
x2 = 0;
xy = 0;
for i = 1:nTrain
   x2 = x2 + X(i,j)^2;
   xy = xy + X(i,j)*y(i);
end
w = xy/x2;

model.name = 'Regrees on One';
model.selectedFeature = selectedFeature;
model.fakeParam = fakeParam;
model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
    [nTest,nFeatures] = size(Xhat);
    w = model.w;
    j = model.selectedFeature;
    
    yhat = zeros(nTest,1);
    for i = 1:nTest
        if (strcmp(model.fakeParam, 'winning-value'))
            yhat(i) = Xhat(i,j)*w;
        else
            yhat(i) = rand() - 0.5;
            if (i > 1)
                yhat(i) = yhat(i-1) + yhat(i);
            end
        end
    end
end