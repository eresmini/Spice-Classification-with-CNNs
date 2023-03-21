# Spices-Dataset

This project involved using convolutional neural networks (CNNs) to identify cooking spices by color and texture.

## Data
I created my own dataset for this problem. At the time of this project, an open-source image dataset of spices did not exist on the Internet.
I used the following cooking spices from my kitchen:
- Allspice
- Alum
- Baking Powder
- Baking Soda
- Basil
- Cayenne Pepper
- Chili Powder
- Cilantro
- Cinnamon
- Cloves
- Coriander
- Cornstarch
- Cream of Tartar
- Crushed Red Pepper
- Cumin
- Garlic Powder
- Ginger
- Granulated Sugar
- Ground White Pepper
- Iodized Salt
- Kosher Salt
- Marjoram
- Minced Garlic
- Mustard
- Nutmeg
- Oregano
- Powdered Sugar
- Sea Salt
- Smoked Paprika
- Tarragon Leaves
- Tumeric

I created a home setup to take the images. The setup consisted of two lamps to prevent shadows, stacked books acting as a makeshift tripod, and sketch paper marking where the equipment and spices should be placed for consistency. I used my iPhone 12 to take the images.
![Picture1](https://user-images.githubusercontent.com/70169642/226624492-f96bfc5a-e63b-4f48-9436-c34f3afdbb4f.jpg)

In total, I took 25 images of each spice. However, these images were high quality and too large to quickly run through a neural network, so I used an image cropper to create 256x256 pixel images. (Code for image cropper from Source: https://stackoverflow.com/questions/53501331/crop-entire-image-with-the-same-cropping-size-with-pil-in-python.) This resulted in approx. 100 images from each original image, with a total of 93,000 images.
![image](https://user-images.githubusercontent.com/70169642/226625840-728fd69c-8992-4f52-9c5b-f246335962f8.png)

## CNN Model
Python - Keras/Tensorflow
To start, I rescaled the images to normalize the RBG values. The model consisted of 4 layers of alternating Conv2D and MaxPooling2D, followed by a flatten layer, one desnse layer with relu activation, and a second dense layer with softmax activation. Adding additional layers did little to change the accuracy of the model, so I opted for fewer layers to help speed up the model fitting time. When compiling, I used the adam optimizer, sparse categorical crossentropy for loss, and had the model return accuracy metrics. The model was run the model on 30 epochs.

## Model Metrics
The model’s predictions show a lower accuracy (5-10 points lower) than the training model, which could suggest overfitting. But overall, the model appeared to perform well. Regardless of changes to the model build, I couldn’t achieve an accuracy higher than 85%.
![image](https://user-images.githubusercontent.com/70169642/226628986-833f6c01-9fa5-442b-8955-a3508f47d659.png)
