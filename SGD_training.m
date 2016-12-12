function[training_error, validation_error, accuracy, theta_best, ...
    break_point] = SGD_training(training_images, training_labels,...
    validation_images, validation_labels, testing_images, testing_labels,...
    LS, theta, delta_weight_prev, epochs)

%this function trains the neural network

samples_num = size(training_images,1);
NL = size(LS, 2);    %num of layers including input and output
learning_rate = 0.1;
RP = 0.8;   %regularization parameter
%Y = zeros(samples_num, LS(NL));
patience = 150; %number of epochs to skip for early stopping
error_min_patience = Inf;
%window_error = 0;   %error in the window to check improvement in validation

num_of_labels = 10;
D = zeros(num_of_labels, samples_num);  
%D has desired response such that for any label, the bit for its index is 1
%in 10 digit binary vector D(sample_number,:) for that sample
for l = 1:num_of_labels - 1
    D(l, :) = training_labels == l;
end;
D(10, :) = training_labels == 0;
%epochs = 150;
%e = zeros(epochs,1);
batch_size = 100;
sample_set = zeros(batch_size, 1);
training_error = zeros(epochs, 1);
%mean_training_error = zeros(epochs, 1);
validation_error = zeros(epochs, 1);
%mean_validation_error = zeros(epochs, 1);
accuracy = zeros(epochs, 1);
for epoch = 1:epochs
    for i = 1:batch_size
        sample_set(i,1) = ceil(rand(1)*size(training_images,1));
        a = (fwd_propagation(training_images(sample_set(i),:), theta, ...
            NL, LS)).';
    %changed a to form (units_in_layer by layer_number)
        [theta, delta_weight_prev] = bwd_propagation(a, theta, ...
            delta_weight_prev, D(:, sample_set(i)), learning_rate, RP, ...
            NL, LS);
    end;
    
    %error = 0;
    all_layers_out = forward_prop(training_images(sample_set, :), theta,...
        NL, LS);
    Y = all_layers_out(1:LS(NL), :, NL);
    error = (Y - D(:, sample_set)).^2;
    training_error(epoch, 1) = sum(sum(error,1))/batch_size;
    validation_error(epoch, 1) = validate(theta, validation_images, ...
        validation_labels, NL, LS);
    accuracy(epoch, 1) = predict(theta, testing_images, testing_labels, ...
        NL, LS);
    if epoch >= patience
        if validation_error(epoch,1) <= error_min_patience
            error_min_patience = validation_error(epoch, 1);
            error_min_index = epoch;
            theta_best = theta;
        end;
        if epoch > error_min_index + 20    %taking window of 10 epochs
            break_point = epoch;
            break;
            %window_error = sum(validation_error(error_min_index:epoch));
        %else
         %   window_mean = 
        end;
    end;
    break_point = epochs;
end;

training_error = training_error(1:break_point);
validation_error = validation_error(1:break_point);
accuracy = accuracy(1:break_point);
figure();
hold on;
plot(1:break_point, training_error, 'LineWidth', 1.5);
plot(1:break_point, validation_error, 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Mean Square Error');
legend('Training Error', 'Validation Error');
title(['Performance for a batch size of ' num2str(batch_size)]);
set(gca, 'fontsize', 23);

figure();
hold on;
plot(1:break_point, accuracy, 'LineWidth', 1.5);
xlabel('Epochs');
ylabel('Accuracy %');
title(['Accuracy percentage curve for batch size of ' num2str(batch_size)]);
set(gca, 'fontsize', 23);
