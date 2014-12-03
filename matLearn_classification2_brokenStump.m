function [model] = matLearn_classification2_brokenStump(X,y,options)
% matLearn_classification2_brokenStump(X,y,options)
%
% Description:
%   - Finds the best threshold across all features
%
% Options:
%   - None
%
% Authors:
% 	- Mark Schmidt (2014)

[nTrain,nFeatures] = size(X);

minErr = inf;
minVar = 0;
for j = 1:nFeatures
    thresholds = [sort(unique(X(:,j)));max(X(:,j))+eps];
    
    for t = thresholds'
        err = sum(X(y==-1,j) < t) + sum(X(y==1,j) >= t);
        
        if err < minErr
            minErr = err;
            selectedFeature = j;
            minThreshold = t;
        end
    end
end

model.name = 'Broken Stump';
model.selectedFeature = selectedFeature;
model.threshold = minThreshold;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
j = model.selectedFeature;
threshold = model.threshold;

yhat = Xhat(:,j) < threshold;
yhat = sign(yhat-.5);
end

