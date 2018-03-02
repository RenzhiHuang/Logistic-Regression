function predict=y_predict(X,w)
predict = 1./(1+exp(-X * w));
predict(predict>1/2)=1;
predict(predict<=1/2)=-1;
end

