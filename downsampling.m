function[training_images, validation_images, testing_images] = ...
    downsampling(training_images, validation_images, testing_images,...
    downsampling_factor)

training_images = (downsample(training_images.',downsampling_factor)).';
validation_images = (downsample(validation_images.',downsampling_factor)).';
testing_images = (downsample(testing_images.',downsampling_factor)).';