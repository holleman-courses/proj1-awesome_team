# Data Collection
- Decided to use the smallest resolution available: 176*144 and grayscale
- Took 900 pictures of pencils and not pencils
- Negative class includes pictures of a screwdriver and a wrench
- Wrote a script to classify the images into folders
- Used the `image_dataset_from_directory` function to load the data

# Model Architecture
- Tried to create as deep a resnet as possible while keeping parameter count below certain threshold
- How to find threshold
    - Arduino has 1MB of flash RAM
    - Subtract 176*144 bytes for image
    - Assuming each parameter is 1 byte, divide by 8 to get the number of parameters
    - This gives us (1MB - 176*144) / 8 = 127,957 parameters
    - We can use this as a rough estimate for the number of parameters we can use
- Found a solid architecture with 9 resblocks and 3 dense layers that uses 114529 params
- Hopefully this will work