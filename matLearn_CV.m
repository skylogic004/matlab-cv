function [bestModel, bestParamValue, bestError, validationErrorLog] = matLearn_CV(X, y, options)
    %{  
     Description:
      - This computes the "best" hyper-parameter using cross-validation for
        classification and regression problems
        (e.g. nHidden with multi-layer perceptron)

     Options:
       - model: the learning algorithm.

       - modelOptions: options you want to pass through to the learning 
                       algorithm

       - nFolds: number of folds, default 5.
      
       - paramName: name of parameter to optimize over.

       - paramValues: list of values to grid search against.
                      e.g. [1, 10, 100, 1000]
                      e.g. {'apples', 'oranges', 'bananas'}

       - loss: name of the loss function used to compute the
               cross-validation score. Can be one of:
               'square error', 'zero-one loss', 'absolute error'

       - earlyStop: stop grid search when the error starts increasing after 
                    it has decreased at least once, default false.

       - shuffle: whether to shuffle the data, default false

       - leaveOneOut: Each sample is used once as a validation set
                      (singleton) while the remaining samples form the 
                      training set.
                      This is equivalent to setting nFolds to the number of
                      samples in the data.

     Output:
       - bestModel: the model with the parameter that achieved the best
                    score. 
       - bestParamValue: the best parameter value found.
       - bestError: the average validation error achieved with the best
                    parameter value.
       - validationErrorLog: struct containing:
         * paramValues: the parameter values list as used by CV (sorted)
         * errorValues: the average validation error corresponding to the
                        parameter values

     Authors:
        - Issam Laradji: issam.laradji@gmail.com
        - Matthew Dirks: http://www.cs.ubc.ca/~mcdirks/
    %}

    X_copy = X;
    y_copy = y;
    validationErrorLog = [];

    % Default values
    [model, modelOptions, nFolds, paramName, paramValues, ...
        loss, shuffle, earlyStop, leaveOneOut] ...
           = myProcessOptions(options,  ...
            'model', NaN,               ...
            'modelOptions', [],         ...
            'nFolds', 5,                ...
            'paramName', NaN,           ...
            'paramValues', NaN,         ...
            'loss', 'square error',     ...
            'shuffle', false,           ...
            'earlyStop', false,         ...
            'leaveOneOut', false        );

    % Default return values
    bestModel = NaN;
    bestParamValue = NaN;
    bestError = NaN;

    % Verify mandatory arguments
    if ~strcmp(class(model), 'function_handle')
        fprintf('ERROR: model must be specified, and must be a function handle.\n');
        return;
    end

    if isnan(paramName)
        fprintf('ERROR: paramName is a mandatory options which was not specified.\n');
        return;
    end
    if ~strcmp(class(paramName), 'char')
        fprintf('ERROR: paramName must be a single string.\n');
        return;
    end
    if isscalar(paramValues)
        if isnan(paramValues)
            fprintf('ERROR: paramValues is a mandatory options which was not specified.\n');
            return;
        end     
        fprintf('ERROR: paramValues must be a vector of more than 1 value.\n');
        return;
    end     

    % Verify valid nFolds option
    if nFolds > size(X,1)
        fprintf('WARNING: nFolds (%d) must not exceed number of samples in the data (%d), using %d for nFolds instead.\n', nFolds, size(X,1), size(X,1));
        nFolds = size(X,1);
    end
    if nFolds < 2
        fprintf('WARNING: nFolds (%d) must be greater than 1, using 2 for nFolds instead.\n', nFolds);
        nFolds = 2;
    end

    % Leave-one-out is a special case of k-fold CV
    % where nFolds = number of samples
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
    else % Default if user provides invalid option
        fprintf('WARNING: invalid loss function name. loss must be one of "square error", "zero-one loss", or "absolute error"\n');
        lossFunction = @square_error;
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
    validationErrorLog.errorValues = [];

    for paramIndex = 1:length(paramValues)
        modelOptions.(options.paramName) = paramValues(paramIndex);
        validationErrors = zeros(nFolds,1);

        % Compute the accumulated score over the folds
        for fold = 1:nFolds
            % Split dataset into training and validation sets
            [Xtrain, ytrain, Xval, yval] = foldData(X, y, nFolds, fold);
            
            % Train model
            trainedModel = model(Xtrain, ytrain, modelOptions);
            
            % Predict y
            yhat = trainedModel.predict(trainedModel, Xval);
            
            % Record error for this fold
            validationErrors(fold) = lossFunction(yhat, yval);  
        end

        avgValidationError = mean(validationErrors);
        validationErrorLog.errorValues(paramIndex) = avgValidationError;
        
        % Naive early stop method to prune parameter search values
        if earlyStop && paramIndex > 1
            % Check if error is decreasing
            if prevValidationError > avgValidationError
                isDecreasedPrev = true;
            end
            % Check if error is increasing, 
            % having decreased at least once previously
            if prevValidationError < avgValidationError && isDecreasedPrev            
                break;
            end
        end
        prevValidationError = avgValidationError;
        
        % Check if new validation error beats previous best
        if avgValidationError < bestError
            bestError = avgValidationError;
            % bestModel = trainedModel;
            
            if iscell(paramValues)
                bestParamValue = paramValues{paramIndex};
            else
                bestParamValue = paramValues(paramIndex);
            end
        end
    end

    modelOptions.(options.paramName) = bestParamValue;
    bestModel = model(X_copy, y_copy, modelOptions);

    % Return the set of parameters that were actually tried, and the order they were used.
    validationErrorLog.paramValues = paramValues(1:length(validationErrorLog.errorValues));
end

function [Xtrain, ytrain, Xval, yval] = foldData(X, y, nFolds, fold)
    % nFolds: total number of folds
    % fold: which fold to perform
        
    n = size(y,1);    
    
    % calculate number of samples per fold
    % (except the last fold will take all remaining rows):
    nPerFold = floor(n/nFolds);
    
    % Create validation set
    valStart = (fold-1)*nPerFold + 1;
    if (fold == nFolds)
        valEnd = n;
    else
        valEnd = fold*nPerFold;
    end
    
    yval = y(valStart:valEnd, :);
    Xval = X(valStart:valEnd, :);
    
    % Training set may consist of 2 parts: the part before the validation
    % set, and the part after.
    if (fold == 1)
        trainStart = valEnd + 1;
        trainEnd = n;
        ytrain = y(trainStart:trainEnd, :);
        Xtrain = X(trainStart:trainEnd, :);
    elseif (fold == nFolds)
        trainStart = 1;
        trainEnd = valStart - 1;
        ytrain = y(trainStart:trainEnd, :);
        Xtrain = X(trainStart:trainEnd, :);
    else % expect 2 parts, because fold must be in the middle
        trainStartA = 1;
        trainEndA = valStart - 1;
        trainStartB = valEnd + 1;
        trainEndB = n;
        
        ytrain = [
            y(trainStartA:trainEndA, :);
            y(trainStartB:trainEndB, :)
        ];
        Xtrain = [
            X(trainStartA:trainEndA, :);
            X(trainStartB:trainEndB, :)
        ];
    end
end

function [error] = square_error(yhat, y)
    % Also known as: average L2^2, or MSE
    error = mean(mean((yhat - y).^2));
end

function [error] = abs_error(yhat, y)
    % Also known as: average L1 norm
    error = mean(mean(abs(yhat - y)));
end

function [error] = zero_one_loss(yhat, y)
    error = mean(mean((yhat ~= y)));
end
