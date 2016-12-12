function[a] = forward_prop(images, theta, NL, LS)

%NL --> number of layers
N = size(images,1);    %num of samples
a = zeros(max(LS)+1, N, NL);    %layer_size by num_of_samples by layer

for layer = 1:NL-1  %add bias unit to each layer
    a(LS(layer)+1,:,layer) = 1;
end;

a(1:LS(1),1:N,1) = images.';
for j = 2:NL
    %z is of the form num_of_samples by units in jth layer
    z = (theta(1:LS(j-1)+1, 1:LS(j), j-1).' * a(1:LS(j-1)+1, :, j-1));
    a(1:LS(j),1:N, j) = sigmoid(z);
end;