function [model] = matLearn_regression_KNN_Dirks(X,y,options)
% matLearn_regression_KNN_Dirks(X,y,options)
%
% Description:
%   - Performs KNN, as from class, but for regression. Built for the
%   purpose of testing matLearn_CV.m
%
% Options:
%   - k: Number of nearest neighbors to use in KNN
%
% Authors:
% 	- Matthew Dirks: http://www.cs.ubc.ca/~mcdirks/

[...
    k, ...
    distFunc, ...
    regressionScoringFunc, ...
    model.weightAvg_power, ...
    model.weightAvg_func, ...
    ] = myProcessOptions(options, ...
    'k',1, ...
    'distFunc', 'cosine', ...
    'regressionScoringFunc', 1, ...
    'weightAvg_power', 1, ... % should be int >= 1, only used in weighted average
    'weightAvg_func', @mean ...
    );

if (regressionScoringFunc == 1)
    regressionScoringFunc = @scoring_mean;
elseif (regressionScoringFunc == 2)
    regressionScoringFunc = @scoring_median;
else
    regressionScoringFunc = @scoring_weightedAverage;
end

model.name = 'KNN by Matthew Dirks';
model.k = k;
model.distFunc = distFunc;
model.regressionScoringFunc = regressionScoringFunc;
model.X = X;
model.y = y;
model.predict = @predict;
end

function [yHat] = predict(model,XHat)
    k = model.k;
    distFunc = model.distFunc;
    X = model.X;
    y = model.y;
    
    [N,P] = size(X);
    [T,D] = size(XHat);
    
%     distances = X.^2*ones(P,T) + ones(N,P)*(XHat').^2 - 2*X*XHat';
    distances = pdist2(X, XHat, distFunc);
    [sortedDistances, sortedIndices] = sort(distances,1); %sort(distances,2);

    sortedNeighbors = y( sortedIndices(1:k, :) ); %indices(:, 1:k)
    if (size(sortedNeighbors,2) == 1) % this means k = 1, so no need to average
        yHat = sortedNeighbors;
    else
%         yHat = mean(yVotes,1)';
        yHat = model.regressionScoringFunc(sortedNeighbors, sortedDistances(1:k, :), model)';
    end
end

function [yHat] = scoring_mean(neighbors, ~, ~)
    yHat = mean(neighbors, 1);
end
function [yHat] = scoring_median(neighbors, ~, ~)
    yHat = median(neighbors, 1);
end
function [yHat] = scoring_weightedAverage(neighbors, distances, model)
    %continuous targets, predicted target is the weighted average of
    %targets for the k-nearest neighbors. The weights are proportional to
    %the inverse of the distance from each k-neighbor to the query point.
    
    % neighbors and distances have similar structure:
    % 1st dimension: size is number of neighbors (k)
    % 2nd dimension: size is number of total examples
    % ie. each column contains that example's k closest neighbors
    
    weights = distances.^-model.weightAvg_power; % inverse
    zs = sum(weights, 1); %normalizing constants
        
    weightedAvgs = sum((weights .* neighbors),1) ./ zs;
    yHatTest = model.weightAvg_func(neighbors, 1);
    yHat = weightedAvgs;    
end