function [bestModel] = matLearn_CV(X, y, options)
    %{  
     Description:
      - This computes the "best" hyper-parameter using cross-validation for classification 
        and regression problems (e.g., nHidden with multi-layer perceptron)
    
     Options:
       - model: the learning algorithm.
 
       - nFolds: number of folds, default 5.
      
       - paramName: name of parameter to optimize over.

       - paramValues: list of values to grid search against.

       - lossFunction: the loss function used to compute the
                       cross-validation score

       - shuffle: whether to shuffle the data, default false
    
     Output:
       - bestModel: the model with the parameter that achieved the best
                    score.

     Authors:
        - Issam Laradji, Matt Dirks (2014)
    %}
    % Default values
    [model, nFolds, paramName, paramValues, loss, shuffle] ...
           = myProcessOptions(options, 'model', NaN, 'nFolds', 2, ...
            'paramName', NaN, 'paramValues', NaN, 'loss', ...
            'square error', 'shuffle', false);

    % Set loss function
    if strcmp(loss, 'square error')
        lossFunction = @square_error;
    elseif strcmp(loss, 'zero one loss')
        lossFunction = @zero_one_loss;
    end
    
    % Shuffle dataset
    if shuffle
        randIndices = randperm(length(X));
        X = X(randIndices, :);
        y = y(randIndices, :);
    end

    % Make sure the param values are sorted
    paramValues = sort(paramValues);
    minError = inf;
    
    for j = 1:length(paramValues)
        subOptions.(options.paramName) = paramValues(j);
        validationError = 0;
        
        % Compute the accumulated score over the folds
        for fold = 1:nFolds
            % Split dataset into training and validation sets
            [Xtrain, ytrain, Xval, yval] = foldData(X, y, nFolds, fold);
            
            % Train model
            trainedModel = model(Xtrain, ytrain, subOptions);
            % Predict y
            yhat = trainedModel.predict(trainedModel, Xval);
            
            % Accumulate score
            validationError = validationError + ...
                              lossFunction(yhat, yval);  
        end
        disp(paramValues(j))
        disp(validationError / nFolds);
        disp('OK');
        
        % Compare the minimum loss
        if validationError / nFolds < minError
            minError = validationError;
            bestModel = trainedModel;
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

function [score] = square_error(yhat, y)
    score = mean((yhat - y).^2);
end

function [score] = zero_one_loss(yhat, y)
    score = mean((yhat ~= y));
end
