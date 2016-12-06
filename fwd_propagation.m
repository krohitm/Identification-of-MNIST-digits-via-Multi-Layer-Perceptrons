function[a] = fwd_propagation(training_image, theta, NL, LS)
%we use the ReLU function for forward propagation
%NL --> number of layers
%LS --> size of layers
a(1,:) = training_image;
for j = 2:NL
    a(j - 1, 1:LS(j - 1)+1) = [1 a(j-1,1:LS(j-1))];
    %jth layer
    z = theta(1:LS(j - 1) + 1, 1:LS(j), j-1).' * a(j - 1, 1:LS(j - 1)+1).';
    %activation of jth layer perceptrons
    for k = 1:length(z)
        a(j, 1:LS(j)) = max(0,z(k,1));
    end;
end;