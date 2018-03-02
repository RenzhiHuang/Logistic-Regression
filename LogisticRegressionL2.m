function [w, b] = LogisticRegressionL2(traindata, trainlabels, lambda)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % lambda      - regularization parameter (positive real number)
        
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    
    % Fill in your code here    
    % Consider using the fminunc MATLAB function for solving the L2- regularized logistic regression optimization problem. 
    L = @(w) (1/length(trainlabels))*sum(log2(1+exp(-trainlabels .* traindata*w)))+lambda*(w'*w-w(end)*w(end));  
    w0=zeros(58,1);
    fitresult = fminunc(L,w0);
    w=fitresult(1:57,1:1);
    b=fitresult(58:58,1:1);
end
