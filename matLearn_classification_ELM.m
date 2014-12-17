function [model] = matLearn_classification_ELM(X, y, options)
    % matLearn_classification_ELM_issam(X,y,options)
    %
    % Description:
    %   - Fits a classification model using the extreme learning machine 
    %     (ELM) algorithm for the purpose of testing matLearn_CV.m
    %   
    % Options:
    %   - n_hidden: Number of hidden neurons to use in ELM.
    %   - lambda: Strength of regularization term.
    %
    % Authors:
    % 	- Issam H. Laradji: issam.laradji@gmail.com
    [nTrain, nFeatures] = size(X);
    nClasses = max(y);
    
     % Default values
    [model.nHidden, model.lambda] = myProcessOptions(options,  ...
            'nHidden', 100, 'lambda', 1);

    % Convert multi-class numerical labels to binary
    y_binarized = zeros(size(y, 1), nClasses);
    for i=1:nTrain
        y_binarized(i, y(i)) = 1;
    end
   
    % Randomize input-to-hidden weights and bias between small values
    interval = 1;
    model.w = rand(nFeatures, model.nHidden) * 2 * interval - interval;
    model.b = rand(1, model.nHidden) * 2 * interval - interval;
    
    % Compute activations of the hidden layer
    H = X * model.w + repmat(model.b, nTrain, 1);
    H = tanh(H);
    
    % Use Ridge Regression to solve for the output weights
    model.beta = ((H' * H + model.lambda * speye(model.nHidden)) \ ...
                   H' * y_binarized); 
               
    model.predict   = @predict;
    
function [yhat] = predict(model, X)
    [nTrain, ~] = size(X);

    % Compute activations of the hidden layer
    H = X * model.w + repmat(model.b, nTrain, 1);
    H = tanh(H);

    % Compute output
    scores = H * model.beta;
    % Return the numerical class labels
    [~, yhat] = max(scores,[],2);
    


