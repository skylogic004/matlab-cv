function [bestModel, bestParamValue, bestError] = matLearn_CV(X, y, options)
    %{  
     Description:
      - This computes the "best" hyper-parameter using cross-validation for classification 
        and regression problems (e.g., nHidden with multi-layer perceptron)
    
     Options:
       - model: the learning algorithm.
 
       - nFolds: number of folds, default 5.
      
       - paramName: name of parameter to optimize over.

       - paramValues: list of values to grid search against.
                      e.g. [1, 10, 100, 1000]
                      e.g. {'apples', 'oranges', 'bananas'}

       - lossFunction: the loss function used to compute the
                       cross-validation score.

       - earlyStop (Added value): stop grid search when the error starts 
                                  increases, default false.

       - shuffle: whether to shuffle the data, default false

       - leaveOneOut: Each sample is used once as a validation set
                      (singleton) while the remaining samples form the 
                      training set.
                      This is equivalent to setting nFolds to the number of
                      samples in the data.

     Output:
       - bestModel: the model with the parameter that achieved the best
                    score.

     Authors:
        - Issam Laradji
        - Matthew Dirks: http://www.cs.ubc.ca/~mcdirks/
    %}

    % Default values
    [model, nFolds, paramName, paramValues, loss, shuffle, earlyStop, leaveOneOut] ...
           = myProcessOptions(options,  ...
            'model', NaN,               ...
            'nFolds', 2,                ...
            'paramName', NaN,           ...
            'paramValues', NaN,         ...
            'loss', 'square error',     ...
            'shuffle', false,           ...
            'earlyStop', false,         ...
            'leaveOneOut', false        );
        
    if leaveOneOut
        nFolds = size(X,1);
    end

    % Set loss function
    if strcmp(loss, 'square error')
        lossFunction = @square_error;
    elseif strcmp(loss, 'zero one loss') || strcmp(loss, 'zero-one loss')
        lossFunction = @zero_one_loss;
    elseif strcmp(loss, 'absolute error')
        lossFunction = @abs_error;
    end
    
    % Shuffle dataset
    if shuffle
        randIndices = randperm(length(X));
        X = X(randIndices, :);
        y = y(randIndices, :);
    end

    % Make sure the param values are sorted
    paramValues = sort(paramValues);
    bestError = inf;
    bestParamValue = NaN;
    isDecreasedPrev = false;

    for paramIndex = 1:length(paramValues)
        subOptions.(options.paramName) = paramValues(paramIndex);
        validationErrors = zeros(nFolds,1);

        % Compute the accumulated score over the folds
        for fold = 1:nFolds
            % Split dataset into training and validation sets
            [Xtrain, ytrain, Xval, yval] = foldData(X, y, nFolds, fold);
            
            % Train model
            trainedModel = model(Xtrain, ytrain, subOptions);
            % Predict y
            yhat = trainedModel.predict(trainedModel, Xval);
            
            % Record error for this fold
            validationErrors(fold) = lossFunction(yhat, yval);  
        end

        avgValidationError = mean(validationErrors);

        % disp(paramValues(paramIndex))
        % disp(validationError / nFolds);
        % disp('OK');
        
        % Naive early stop method to prune parameter search values
        if earlyStop && paramIndex > 1
            % Check if error is decreasing
            if prevValidationError > avgValidationError
                isDecreasedPrev = true;
            end
            % Check if error is increasing, having decreased at least once previously
            if prevValidationError < avgValidationError && isDecreasedPrev            
                break;
            end
        end
        prevValidationError = avgValidationError;
        
        % Check if new validation error beats previous best
        if avgValidationError < bestError
            bestError = avgValidationError;
            bestModel = trainedModel;
            
            if iscell(paramValues)
                bestParamValue = paramValues{paramIndex};
            else
                bestParamValue = paramValues(paramIndex);
            end
        end
   end
end

function [Xtrain, ytrain, Xval, yval] = foldData(X, y, nFolds, fold)
    % nFolds: total number of folds
    % fold: which fold to perform
        
    n = size(y,1);    
    nPerFold = floor(n/nFolds); % excluding the last fold, which takes all remaining rows
    
    % Create validation set
    valStart = (fold-1)*nPerFold + 1;
    if (fold == nFolds)
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
    elseif (fold == nFolds)
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

function [error] = square_error(yhat, y)
    % Also known as: average L2^2, or MSE
    error = mean((yhat - y).^2);
end

function [error] = abs_error(yhat, y)
    % Also known as: average L1 norm
    error = mean(abs(yhat - y));
end

function [error] = zero_one_loss(yhat, y)
    error = mean((yhat ~= y));
end
