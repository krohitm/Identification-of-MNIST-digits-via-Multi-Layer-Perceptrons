function[] = runProgram(hidden_layers_sizes)

[training_images, training_labels, validation_images, ...
    validation_labels, testing_images, testing_labels] = loadData();

num_of_input_units = size(training_images,2);
num_of_output_units = 10;

layers_sizes = [num_of_input_units hidden_layers_sizes num_of_output_units];
theta = initializeTheta(layers_sizes);
for epochs = 1:10
    [Y,theta] = training_nn(training_images, training_labels,...
        layer_sizes, theta);
end;