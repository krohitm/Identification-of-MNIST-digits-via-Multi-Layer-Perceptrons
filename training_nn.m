function[Y, theta] = training_nn(training_images, training_labels, LS, theta)
%this function trains the neural network

samples_num = size(training_images,1);
NL = size(LS, 2);    %num of layers including input and output
learning_rate = 0.3;
RP = 0.1;   %regularization parameter
Y = zeros(samples_num, LS(NL));
thetaPrev = theta;
num_of_labels = 10;
%D = zeros(samples_num, num_of_labels);  
%D has desired response such that for any label, the bit for its index is 1
%in 10 digit binary vector D(sample_number,:) for that sample
%for l = 1:num_of_labels - 1
%    D(:,l) = training_labels == l;
%end;
%D(:,10) = training_labels == 0;
D =training_labels;

for i = 1:samples_num
    a = fwd_propagation(training_images(i,:), theta, NL, LS);
    Y(i,1:LS(NL)) = a(NL, 1:LS(NL));
    [theta, thetaPrev] = bwd_propagation(a, theta, thetaPrev, D(i,:), ...
         learning_rate, RP, NL, LS);
    
    %clearvars a;
end;