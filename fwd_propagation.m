function[a] = fwd_propagation(training_image, theta, NL, LS)
%we use the ReLU function for forward propagation
%NL --> number of layers
%LS --> size of layers
a(1,:) = training_image;
for j = 2:NL
    a(j - 1, 1:LS(j - 1)+1) = [a(j-1,1:LS(j-1)) 1];
    %jth layer
    z = a(j - 1, 1:LS(j - 1)+1) * theta(1:LS(j-1)+1, 1:LS(j), j-1);
    a(j, 1:LS(j)) = sigmoid(z);
end;