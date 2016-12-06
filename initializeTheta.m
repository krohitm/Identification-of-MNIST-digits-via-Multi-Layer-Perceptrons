function[theta] = initializeTheta(layers_sizes)
%this function is to initialize theta
%we'll be keeping initial bias weight as 1

num_layers = size(layers_sizes, 2);

theta = zeros(max(layers_sizes), max(layers_sizes(2:length(layers_sizes))) ,num_layers - 1);
for i=1:num_layers - 1
    e = sqrt(6)/sqrt(layers_sizes(1,i) + layers_sizes(i+1));
    theta(1:layers_sizes(1,i) + 1, 1:layers_sizes(1,i+1), i) = ...
        rand(layers_sizes(1,i) + 1, layers_sizes(1,i+1)) * (2 * e) - e;
end;
