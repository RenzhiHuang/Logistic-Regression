function [w, b] = LogisticRegression(traindata, trainlabels)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data    
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    % Fill in your code here    
    % Consider using fminunc MATLAB function for solving the logistic regression optimization problem.
    L = @(w) sum(log(1+exp(-trainlabels .* traindata*w)));  
    w0=zeros(58,1);
    fitresult = fminunc(L,w0);
    w=fitresult(1:57,1:1);
    b=fitresult(58:58,1:1);
end
