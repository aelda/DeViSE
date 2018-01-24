# DeViSE

We implemented the DeViSE network (Frome et al., 2013) that can 
**recognize scenes from tens of thousands of images and show that the semantic information** can be exploited to make predictions about more image labels which are not observed during training. 

## Model
we use two pre-trained models in order to implement the DeViSE .

**As for the Visual model** :
* a pre-trained AlexNet
* SUN3971, a dataset for scene recognition

**As for the Semantic model** :
* Global Vectors forWord Representation (GloVe)
