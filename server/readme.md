### Server side implementations

This folder contains the server side implementation. Each frame of the video is recieved as an array through socket communication. 

 - [x] Detect objects in frame [object-recognition]
 - [ ] Detect action based on the objects found [action-recognition]
 - [ ] Train LSTM with the sequence of features including actions [model]
 - [ ]  Predict the intent [prediction]