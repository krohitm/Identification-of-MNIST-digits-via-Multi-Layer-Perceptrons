function[Y] = predict(X, theta, LS)

samples_num = size(X,1);
NL = size(LS, 2);    %num of layers including input and output

for i = 1:samples_num
    
    a(1,:) = X(i,:);
    for j = 2:NL
        a(j - 1, 1:LS(j - 1)+1) = [1 a(j-1,1:LS(j-1))];
        z = theta(1:LS(j - 1) + 1, 1:LS(j), j-1).' * ...
            a(j - 1, 1:LS(j - 1)+1).';   %jth layer, ith sample
        %activation of jth layer perceptrons
        
        a(j, 1:LS(j)) = sigmoid(z).';
    end;
    
    Y(i,LS(NL)) = a(NL, 1:LS(NL));
    clearvars a;
end;

%E = (1/8)*(sum((D - Y).^2));
Y(Y>=0.5) = 1;
Y(Y<0.5) = 0;