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
    options.model
    [model, nFolds, paramName, paramValues, lossFunction, shuffle] ...
           = myProcessOptions(options, 'model', NaN, 'nFolds', 2, ...
            'paramName', NaN, 'paramValues', NaN, 'lossFunction', ...
            @square_error, 'shuffle', false);
         
    [nTrain, ~] = size(X);
    
    % Shuffle dataset
    randIndices = randperm(length(X));
    X = X(randIndices, :);
    y = y(randIndices, :);

    % Get the fold indices for Xtrain and Xcv
    foldIndices = crossvalind('Kfold', nTrain, nFolds);
    
    % Make sure the param values are sorted
    paramValues = sort(paramValues);
    minError = inf;
    
    for j = 1:length(paramValues)
        subOptions.(options.paramName) = paramValues(j);
        validationError = 0;
        
        % Compute the accumulated score over the folds
        for k = 1:nFolds
            trainIndices =  foldIndices == k;
            cvIndices =  foldIndices ~= k;
            
            % Train model
            trainedModel = model(X(trainIndices, :),y(trainIndices, :), subOptions);
            % Predict y
            yhat = trainedModel.predict(trainedModel, X(cvIndices, :));
            
            % Accumulate score
            validationError = validationError + ...
                              lossFunction(yhat,y(cvIndices));  
        end
        disp(paramValues(j))
        disp(validationError / k);
        disp('OK');
        
        % Compare the minimum loss
        if validationError / k < minError
            minError = validationError;
            bestModel = trainedModel;
        end
   end
     
end
function [score] = square_error(yhat, y)
    score = mean((yhat - y).^2);
end

function [score] = zero_one_loss(yhat, y)
    score = mean((yhat ~= y));
end
