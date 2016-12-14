function[] = runProgram(hidden_layers_sizes)

[training_images, training_labels, validation_images, ...
    validation_labels, testing_images, testing_labels] = loadData();

training_images_bkp = training_images;
testing_images_bkp = testing_images;
validation_images_bkp = validation_images;

[training_images, validation_images, testing_images] = ...
    downsampling(training_images_bkp, validation_images_bkp, ...
    testing_images_bkp, downsampling_factor);

num_of_input_units = size(training_images,2);
num_of_output_units = 10;

layers_sizes = [num_of_input_units hidden_layers_sizes num_of_output_units];
theta = initializeTheta(layers_sizes);
delta_weight_prev = zeros(size(theta,1), size(theta,2), size(theta,3));

[Y, D, theta, delta_weight_prev] = batch_training(...
    training_images,training_labels, validation_images, ...
        validation_labels, layers_sizes, theta, delta_weight_prev, epochs);

[Y, D, theta, delta_weight_prev] = SGD_training(...
    training_images,training_labels, validation_images, ...
        validation_labels, LS, theta, delta_weight_prev, epochs);
NL = length(layers_sizes);  
cf = predict(theta, testing_images, testing_labels, NL, LS);