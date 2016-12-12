function[accuracy, Y] = predict(theta, testing_images, testing_labels, NL, LS)

N = size(testing_images,1);    %num of samples
a = zeros(max(LS)+1, N, NL);    %layer_size by num_of_samples by layer

for layer = 1:NL-1  %add bias unit to each layer
    a(LS(layer)+1,:,layer) = 1;
end;
a(1:LS(1),1:N,1) = testing_images.';

num_of_labels = 10;
for l = 1:num_of_labels - 1
    D(l, :) = testing_labels == l;
end;
D(10, :) = testing_labels == 0;

for j = 2:NL
    %z is of the form num_of_samples by units in jth layer
    z = (theta(1:LS(j-1)+1, 1:LS(j), j-1).' * a(1:LS(j-1)+1, :, j-1));
    a(1:LS(j),1:N, j) = sigmoid(z);
end;

[val idx] = max(a(1:LS(NL), 1:N, NL));
Y = (mod(idx,10)).';
%cf = confusionmat(testing_labels,Y);
check_match = Y == testing_labels;
accuracy = ((sum(check_match))/N)*100;
