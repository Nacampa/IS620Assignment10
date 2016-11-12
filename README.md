# IS620Assignment10
Movie Review Classification

Assignment

I choose movie reviews and ratings.
The top 2000 most frequent words were selected. 

Features were trained and tested using both a 90/10 percent model and a tested
on the first 100 and train on remainder model.

The functions used to train and test features: document_features,  document_features_all  and document_features_sentiment.

Features were also trained and tested with the stop words remove:  document_festures_all
and document_festures_sentiment.

document_features_sentiment adds the word sentiment value (ranging from -5 to +5) using  AFINN word sentiment
data and  to a (-1) or (1) using Harvard’s Inquire basic data. For Harvard’s data, the positive word polarity is converted
to (1) and negative word polarity to (-1) within the program.

A table showing each method and classification accuracy displays 
