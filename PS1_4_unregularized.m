%% Logistic Regression
% Train data size
m_train=250;
m_test=4351;
% Input the sample matrix and partition it into data 'x' and label 'y'
% Add an extra column to x to accommondate the bias term
x_train=train100(1:m_train,1:57);
x_train(:,58)=ones(m_train,1);
y_train=train100(1:m_train,58:58);
x_test=test(1:m_test,1:57);
x_test(:,58)=ones(m_test,1);
y_test=test(1:m_test,58:58);
% Call the function LogisticRegression to get [w,b]
[w,b]=LogisticRegression(x_train,y_train);
% Call the function y_predict to get the predicted labels
y_predict_train = y_predict(x_train,[w;b]);
y_predict_test = y_predict(x_test,[w;b]);
% Call the function classification_error to get the train error and test
% error
er_train = classification_error(y_predict_train,y_train);
er_test = classification_error(y_predict_test,y_test);

%% Sanity Checking

% Input the train labels in {0,1}
glm_y_train=y_train;
glm_y_train(find(glm_y_train==-1))=0;

%Call the function glmfit to do the logit regression and check the train
%error & test error
glm=glmfit(x_train,glm_y_train,'binomial','link','logit','constant','off');
glm_y_predict_train = y_predict(x_train,glm);
glm_y_predict_test = y_predict(x_test,glm);
glm_er_train = classification_error(glm_y_predict_train,y_train);
glm_er_test = classification_error(glm_y_predict_test,y_test);