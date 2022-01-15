# Arduino-Speech-Recognition-Testing


In this project, we are going to use machine learning to train a speech recognition model using Edge Impulse Studio with three commands i.e. ‘LIGHT ON’, ‘LIGHT OFF’, and ‘NOISE’. Edge Impulse is an online machine learning platform that enables developers to create the next generation of intelligent device solutions with embedded Machine Learning.

*****************************************************************************

# Creating the Dataset for Arduino Speech Recognition

Here Edge Impulse Studio is used to train our Speech Recognition model. Training a model on Edge Impulse Studio is similar to training machine learning models on other machine learning frameworks. For training, a machine learning model's first step is to collect a dataset that has the samples of data that we would like to be able to recognize.

As our goal is to control an LED with our voice command, we will need to collect voice samples for all the commands and noise so that it can distinguish between voice commands and other Noises.

We will create a dataset with three classes “LED ON”, “LED OFF” and “noise”. To create a dataset, create an Edge Impulse account, verify your account and then start a new project. You can load the samples by using your mobile, your Arduino board or you can import a dataset into your edge impulse account. The easiest way to load the samples into your account is by using your mobile phone. For that connect the mobile with Edge Impulse.

To connect Mobile phone click on ‘Devices’ and then click on ‘Connect a New Device’.

Now in the next window click on ‘Use your Mobile Phone’, and a QR code will appear. Scan the QR code with your Mobile Phone or enter the URL given on QR code.

This will connect your phone with Edge Impulse studio.

With your phone connected with Edge Impulse Studio, you can now load your samples. To load the samples click on ‘Data acquisition’. Now on the Data acquisition page enter the label name, select the microphone as a sensor, and enter the sample length. Click on ‘Start sampling’, your device will capture a 2 Sec sample. Record a total of 10 to 12 voice samples in different conditions.

After uploading the samples for the first class now set the change the label and collect the samples for ‘light off’ and ‘noise’ class.

These samples are for Training the module, in the next steps, we will collect the Test Data. Test data should be at least 30% of training data, so collect the 4 samples of ‘noise’ and 4 to 5 samples for ‘light on’ and ‘light off’.

As our dataset is ready, now we can create an impulse for the data. For that go to ‘Create impulse’ page. Change the default settings of a 1000 ms Window size to 1200ms and 500 ms Window increase to 50ms. This means our data will be processed 1.2 s at a time, starting each 58ms.

Now on ‘Create impulse’ page click on ‘Add a processing block’. In the next window select the Audio (MFCC) block. After that click on ‘Add a learning block’ and select the Neural Network (Keras) block. Then click on ‘Save Impulse’.

In the next step go to the MFCC page and then click on ‘Generate Features’. It will generate MFCC blocks for all of our windows of audio.

After that go to the ‘NN Classifier’ page and click on the three dots on the upper right corner of the ‘Neural Network settings’  and select ‘Switch to Keras (expert) mode’.

Replace the original with the following code and change the ‘Minimum confidence rating’ to ‘0.70’. Then click on the ‘Start training’ button. It will start training your model.

********************************************

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, InputLayer, Dropout, Flatten, Reshape, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.constraints import MaxNorm

model = Sequential()

model.add(InputLayer(input_shape=(X_train.shape[1], ), name='x_input'))

model.add(Reshape((int(X_train.shape[1] / 13), 13, 1), input_shape=(X_train.shape[1], )))

model.add(Conv2D(10, kernel_size=5, activation='relu', padding='same', kernel_constraint=MaxNorm(3)))

model.add(AveragePooling2D(pool_size=2, padding='same'))

model.add(Conv2D(5, kernel_size=5, activation='relu', padding='same', kernel_constraint=MaxNorm(3)))

model.add(AveragePooling2D(pool_size=2, padding='same'))

model.add(Flatten())

model.add(Dense(classes, activation='softmax', name='y_pred', kernel_constraint=MaxNorm(3)))

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=9, validation_data=(X_test, Y_test), verbose=2)


**************************************************

After training the model it will show the training performance. 

You can increase your model's performance by creating a vast dataset.

Now as our Speech Recognition model is ready, we will deploy this model as Arduino library. Before downloading the model as a library you can test the performance by going to the ‘Live Classification’ page. The Live classification feature allows you to test the model both with the existing testing data that came with the dataset or by streaming audio data from your mobile phone.

To test the data with your phone, choose ‘Switch to Classification Mode’ on your phone.

Now to download the model as Arduino Library, go to the ‘Deployment’ page and select ‘Arduino Library’. Now scroll down and click on ‘Build’ to start the process. This will build an Arduino library for your project.

Now add the library in your Arduino IDE. For that open the Arduino IDE and then click on Sketch > Include Library > Add.ZIP library 

Then, load an example by going to File > Examples > Your project name - Edge Impulse > nano_ble33_sense_microphone

# Arduino Code for Arduino Voice Recognition

Here some changes have been made to control LED with the voice commands.

We are making some changes in the void loop() where it is printing the probability of commands. In the original code, it is printing all the labels and their values together.

**************************************************

for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {

ei_printf("    %s: %.5f\n", result.classification[ix].label, result.classification[ix].value);

   }
   
************************************************** 

To control the LED we have to save all the command probabilities in three different variables so that we can put conditional statements on them. So according to the new code if the probability of ‘light on’ command is more than 0.50 then it will turn on the LED and if the probability of ‘light off’ command is more than 0.50 than it will turn off the LED.

************************************************** 

for (size_t ix = 2; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {

    noise = result.classification[ix].value;
   
    
    Serial.println("Noise: ");
    
    Serial.println(noise);
    
    } 
    
    for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix--) {
    
    lightoff = result.classification[ix].value;
    
    Serial.println("Light Off: ");
    
   
    Serial.print(lightoff);
    
    }
    
    lighton = 1- (noise +lightoff);
    
    Serial.println("Light ON: ");
    
    Serial.print(lighton);
    
    if (lighton > 0.50)
    
    {
    
      digitalWrite(led, HIGH);
      
    }
    
    if (lightoff > 0.50){
    
      digitalWrite(led, LOW);
      
    }

************************************************** 
   
After making the changes, upload the code into your Arduino. Open the serial monitor at 115200 baud.
   
   
# This is how you can build speech recognition using Arduino and give commands to operate the device
