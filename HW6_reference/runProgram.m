function[X, initial_theta, D, layers_sizes, Y, theta, E] = ...
    runProgram(hidden_layers_sizes)

%theta is in the form (input_layer_size : output_layer_size : input_layer)
%here input layer includes bias as well
%X is in the form(samples : features) with an additional bias 1 added below
[X, D] = trainingData();

input_layer_size = size(X,2);
output_size = 1;

%sizes of all layers i.e. input, hidden and output, excluding bias
layers_sizes = [input_layer_size hidden_layers_sizes output_size]; 

initial_theta = initializeTheta(layers_sizes);
theta = initial_theta;
for i = 1:2000
    [Y, theta, E(i,1)] = FWDBWD(theta, X, D, layers_sizes);
end;

plot(1:size(E,1), E(:,1));
xlabel('Epochs');
ylabel('Mean Square Error');
title('Mean Square Error against epochs');