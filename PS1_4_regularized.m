%%Cross validation
% Train data size
m_train=200;
m_test=50;
for i=0:7
    lambda = 10^(i-7);
    train = cvtrain_fold5;
    test = cvtest_fold5;
    % Input the sample matrix and partition it into data 'x' and label 'y'
    % Add an extra column to x to accommondate the bias term
    x_train=train(1:m_train,1:57);
    x_train(:,58)=ones(m_train,1);
    y_train=train(1:m_train,58:58);
    x_test=test(1:m_test,1:57);
    x_test(:,58)=ones(m_test,1);
    y_test=test(1:m_test,58:58);
    % Call the function LogisticRegression to get [w,b]
    [w,b]=LogisticRegressionL2(x_train,y_train,lambda);
    % Call the function y_predict to get the predicted labels
    y_predict_train = y_predict(x_train,[w;b]);
    y_predict_test = y_predict(x_test,[w;b]);
    % Call the function classification_error to get the train error and test
    % error
    er_train = classification_error(y_predict_train,y_train);
    er_test = classification_error(y_predict_test,y_test);
    cvtobeplot(i+1,6) = er_test;
end


