function[Y, D, theta, delta_weight_prev] = batch_training(...
    training_images,training_labels, validation_images, ...
        validation_labels, LS, theta, delta_weight_prev, epochs)

samples_num = size(training_images,1);
NL = size(LS, 2);    %num of layers including input and output
learning_rate = 0.01;
RP = 0.8;   %regularization parameter
num_of_labels = 10;
D = zeros(num_of_labels, samples_num);  
%D has desired response such that for any label, the bit for its index is 1
%in 10 digit binary vector D(sample_number,:) for that sample
for l = 1:num_of_labels - 1
    D(l,:) = training_labels == l;
end;
D(10,:) = training_labels == 0;

%D = training_labels.';
training_error = zeros(epochs,1);
validation_error = zeros(epochs,1);
for epoch=1:epochs
    a = forward_prop(training_images, theta, NL, LS);
    [theta, delta_weight_prev] = back_prop(a, D, theta, delta_weight_prev, ...
        learning_rate, RP, NL, LS);
    %all_layers_out = forward_prop(training_images, theta, NL, LS);
    Y = a(1:LS(NL), :, NL);
    error = (Y - D).^2;
    training_error(epoch, 1) = sum(sum(error,1))/samples_num;
    validation_error(epoch, 1) = validate(theta, validation_images, ...
        validation_labels, NL, LS);
end;

figure();
hold on;
plot(1:length(training_error), training_error, 'LineWidth', 1.5);
plot(1:length(validation_error), validation_error, 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Mean Square Error');
legend('Training Error','Validation Error');
title('Mean Square error rate for batch training ')
set(gca, 'fontsize', 23);