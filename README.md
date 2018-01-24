# DeViSE

We implemented the DeViSE network (Frome et al., 2013) that can 
***recognize scenes from tens of thousands of images and show that the semantic information** can be exploited to make predictions about more image labels which are not observed during training. 
In comparison to the original work, we did some alternations.

## Differences
We changed the original semantic model, a skip-gramWord2Vec model, into GloVe (Pennington et al., 2014), an unsupervised learning algorithm
for obtaining vector representations for words.
Unfortunately, the baseline model did not work well as expected.

## Improvement
To improve the model, we fixed some problems and tried some modifications, for example, to change the pre-trained language model and some data processing
