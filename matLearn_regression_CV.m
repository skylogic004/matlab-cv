function [model] = matLearn_regression_CV(modelRef, X, y, options)
% matLearn_regression_CV(modelRef, X, y, options)
%
% Description:
%   Picks the best hyper-parameter (algorithm parameter) value using
%   X and y (which should not include your test set).
%   X & y, is split into new training and validation sets based on the
%   number of folds, k, specified. Algorithm parameter values are iterated
%   exhaustively (brute-force search). Scoring is average absolute test
%   error.
%
% Options:
%   - paramStruct: (struct) Defines the algorithm parameter to optimize and
%     the values that should be searched. For example:
%         paramStruct = [];
%         paramStruct.name = 'stepSize';
%         paramStruct.values = [.0001 .001 .01 .1 0.4 0.5];
%   - mode: (string) Either "k-fold" (default) or "leave-one-out"
%   - k: (int) The number of folds to use in k-fold cross validation
%     (ignored when mode is "leave-one-out").
%   - p: (int) Defines which error function to use in evaluating a model.
%     If p=1 (default), uses average absolute error (average L1 norm).
%     If p=2, uses average squared-error (average L_2^2, or MSE).
%   uses average 
%   - shuffleData: (boolean) If set to true, will randomly order the rows
%      of X and y, while ensuring that rows in y still correspond to rows
%      in X.
%
% Authors:
% 	- Matthew Dirks (2014) http://www.cs.ubc.ca/~mcdirks/
%
    [mode, k, paramStruct, shuffleData, p, verbose] = myProcessOptions(options, 'mode', 'k-fold', 'k',2, 'params', NaN, 'shuffleData', false, 'p', 1, 'verbose', false);

    if (strcmp(mode, 'leave-one-out'))
        k = size(X,1);
    end
    
    % Do shuffle?
    if (shuffleData)
        [X,y] = shuffle(X,y);
    end
    
    % Search over parameter values
    nValues = length(paramStruct.values);
    results = [];
    results.bestErr = Inf;
    results.bestParam = 0;
    
    for i = 1:nValues
        if (verbose)
            fprintf('Trying parameter #%d, value = %s  ', i, paramStruct.values(i));
        end
        modelOptions = [];
        modelOptions.selectedFeature = 1;
        modelOptions.(paramStruct.name) = paramStruct.values(i);
        
        [tmpErr] = doCV(X, y, modelRef, modelOptions, k, p);
        
        if (verbose)
            fprintf('--> avg error = %0.3f\n', tmpErr);
        end
        
        if (tmpErr < results.bestErr)
            results.bestErr = tmpErr;
            results.bestParam = i;
        end
    end
    
    if (verbose)
        fprintf('Best error: %0.3f, Best param: #%d with value %s\n', results.bestErr, results.bestParam, paramStruct.values(results.bestParam));
    end
    
    % Create model with best params found
    modelOptions = [];
    modelOptions.selectedFeature = 1;
    modelOptions.(paramStruct.name) = paramStruct.values(results.bestParam);
    [model] = modelRef(X,y,modelOptions);
end

function [Xshuffled, yshuffled] = shuffle(X, y)
    % Shuffle order of data randomly - but making sure that y still
    % corresponds to X.
    
    s = rng; % save seed
    yshuffled=y(randperm(length(y)));
    
    rng(s); % restore seed, so that sort performs exactly as before
    Xshuffled = X(randperm(size(X,1)), :);
end

function [Xtrain,ytrain,Xval,yval] = foldData(X, y, k, fold)
    % k: total number of folds
    % fold: which fold to perform
        
    n = size(y,1);    
    nPerFold = floor(n/k); % excluding the last fold, which takes all remaining rows
    
    % Create validation set
    valStart = (fold-1)*nPerFold + 1;
    if (fold == k)
        valEnd = n;
    else
        valEnd = fold*nPerFold;
    end
    
    yval = y(valStart:valEnd);
    Xval = X(valStart:valEnd, :);
    
    % Training set may consist of 2 parts: the part before the validation
    % set, and the part after.
    if (fold == 1)
        trainStart = valEnd + 1;
        trainEnd = n;
        ytrain = y(trainStart:trainEnd);
        Xtrain = X(trainStart:trainEnd, :);
    elseif (fold == k)
        trainStart = 1;
        trainEnd = valStart - 1;
        ytrain = y(trainStart:trainEnd);
        Xtrain = X(trainStart:trainEnd, :);
    else % expect 2 parts, because fold must be in the middle
        trainStartA = 1;
        trainEndA = valStart - 1;
        trainStartB = valEnd + 1;
        trainEndB = n;
        
        ytrain = [
            y(trainStartA:trainEndA);
            y(trainStartB:trainEndB)
        ];
        Xtrain = [
            X(trainStartA:trainEndA, :);
            X(trainStartB:trainEndB, :)
        ];
    end
end

function [score] = calcScore(y1, y2, p)
    if (p == 1) % avg L1 norm
        score = mean(abs(y2-y1));
    elseif (p == 2) % avg L2^2
        score = mean((y2-y1).^2);
    else
        fprintf('ERROR: Invalid p specified in calcScore function\n');
    end
end

function [avgErr] = doCV(X,y, modelRef, modelOptions, k, p)
    testErrors = zeros(k,1);

    for fold = 1:k
        [Xtrain,ytrain,Xval,yval] = foldData(X, y, k, fold);
        [model] = modelRef(Xtrain,ytrain,modelOptions);

        yhat = model.predict(model,Xval);

        % Measure test error
        testErrors(fold) = calcScore(yval, yhat, p);
        
        %fprintf('%.3f  ',testErrors(fold));
    end
    avgErr = mean(testErrors);
    
    %fprintf('\n---avg = %0.3f\n', avgErr);
end
