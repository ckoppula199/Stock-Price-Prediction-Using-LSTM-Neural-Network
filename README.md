# Stock-Price-Prediction-Using-LSTM-Neural-Network  
  
This program uses a Recurrent Neural Network (LSTM) to try and predict the prices of a companies stocks using the previous 60 days stock prices as input.  
  
In reality using this approcah to predict stock prices isn't a good strategy. Share prices dont tend to solely or even mainly depend on their previous prices and are significantly influenced by things such as the opinions about the stocks held by traders and the companies public image.  
  
The results the RNN produces slightly lags behind the real price values, similar to a movning average. This allows it to minimise the MSE loss funtion and why it appears to be doing a very good job. A possible improvement may be to predict the changes between time steps rather than the price itself.  
  
I completed this as a way of applying my knowledge on RNNs to a project, not as a serious way of predicitng stock prices.
