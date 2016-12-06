function[Y, theta, E] = FWDBWD(theta, training, D, LS)
%finding output by forward propagation 

samples_num = size(X,1);
NL = size(LS, 2);    %num of layers including input and output
learning_rate = 0.3;
RP = 0.1;   %regularization parameter
Y = zeros(samples_num,LS(NL));
delta = zeros(max(LS), NL);   %won't be using delta 1 but kept for ease of convention
thetaPrev = theta;

%forward and back propagation for each sample
for i = 1:samples_num
    a(1,:) = X(i,:);
    for j = 2:NL
        a(j - 1, 1:LS(j - 1)+1) = [1 a(j-1,1:LS(j-1))];
        z = theta(1:LS(j - 1) + 1, 1:LS(j), j-1).' * ...
            a(j - 1, 1:LS(j - 1)+1).';   %jth layer, ith sample
        %activation of jth layer perceptrons
        a(j, 1:LS(j)) = sigmoid(z).';
    end;
    
    %at the end of the above loop, a will be as (layer:neurons)
    Y(i,LS(NL)) = ...
        a(NL, 1:LS(NL));
    j = NL;
    delta(1:LS(j),j) = (D(i, 1:LS(j)) - Y(i,1)) * Y(i,1) * (1 - Y(i,1));
        
    %get delta values for all layers except 1st
    for j = (NL - 1):-1:2
        delta(1:LS(j), j) = (((theta(2:LS(j) + 1, 1:LS(j+1), j)...
            * delta(1:LS(j+1), j+1)).')...
            .*a(j,1:LS(j)).*(1 - a(j,1:LS(j)))).';
    end;
    
    thetaTemp = theta;
    %update weights for all layers
    for j = NL:-1:2
        theta(1:LS(j - 1) + 1, 1:LS(j), j - 1) = ...
            theta(1:LS(j - 1) + 1, 1:LS(j), j - 1) ...
            + (learning_rate * delta(1:LS(j), j) *  ...
            a(j-1, 1:LS(j-1) + 1)).' + ...
            RP * ([zeros(1, LS(j)); theta(2:LS(j - 1) + 1, 1:LS(j), j - 1)] ...
            - [zeros(1, LS(j)); thetaPrev(2:LS(j - 1) + 1, 1:LS(j), j - 1)]);
    end;
    thetaPrev = thetaTemp;
    clearvars a;
end;

E = (1/8)*(sum((D - Y).^2));
Y(Y>=0.5) = 1;  %values above and equal to 0.5 in sigmoid function being classified to class 1
Y(Y<0.5) = 0;