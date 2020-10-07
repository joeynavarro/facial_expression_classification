# Facial Emotion Classification Using Deep Convolutional Neural Networks

![crowd](.\readme_images\crowd.png)

**[Image Sourced From Kairos](https://www.kairos.com/blog/face-detection-explained)**

## Introduction

There are times when we don't recognize our own emotions as we feel them. Usually, a good indicator of our emotional state is the expression we wear on our face. I would like to be able to take an image of a face such as a selfie and use it as a means to recognize the emotional state that the featured person of the image is in based off of their facial expression. In order to accomplish this I will build two convolutional neural network models of known model architecture (VGG19, ResNet18) that will take in flattened image pixel arrays as input data. The pixel arrays will be scaled and split into a training and testing set which will be used by the models built within a framework created utilizing  the PyTorch machine learning library.  The framework will facilitate creation of a model through utilizing training data learning on testing data for a reasonable range of epoch passes. The trained model will then be able to be utilized to predict facial expressions and report the expression predicted.



## Problem Statement

Utilize pre-classified images to train a model for the purpose of facial expression recognition using a deep convolutional neural network.



## Data

The CK+ dataset is an extension of the CK dataset. The original dataset contains 327 labeled facial videos. The dataset used here is composed of the last three frames from each video in the CK+ dataset, therefore it contains a total of 981 facial expressions. 

To learn more about the dataset go **[here](http://www.iainm.com/publications/Lucey2010-The-Extended/paper.pdf)**. To request access to the full CK+ dataset submit **[this](http://www.jeffcohn.net/wp-content/uploads/2020/04/CK-AgreementForm.pdf)** form.



## Models

Two deep convolutional neural network model architectures are used in this project:

### VGG19

This network's name comes from it's founding group at Oxford, namely Visual Geometry Group (VGG). The 19 after the name stands for the number of layers in the network. This network can be used for transfer learning, therefore it may also be used for facial recognition tasks. It's architecture is as follows:

![vgg19_architecture](.\readme_images\vgg19_architecture.jpg)

You may find more information by reading this **[VGG19 Research Paper](https://arxiv.org/pdf/1409.1556.pdf)**.

### Resnet18

This network's name comes from the functions used by the network, namely residual learning functions or Residual learning Network (ResNet).   The 18 after the name stands for the number of layers in the network. It's architecture is as follows:

![resnet18_architecture](.\readme_images\resnet18_architecture.jpg)



You may find more information by reading this **[ResNet18 Research Paper](https://arxiv.org/pdf/1512.03385.pdf)**.



## Project Workflow

The lifeblood of this project lives within the model_src directory. There you will find the model_executor.ipynb Jupyter Notebook and a model_executory.py Python script. Both are equivalent to creating the model. The model_executor file contains an image sorter, train function, test function, image array creator, transformers, and neural network executor for VGG19 and for ResNet18, it also creates trained models, and visualizations. The trained models are saved in the model_checkpoints directory in the root directory and the visualizations are saved in the model_visualizations directory also in the root directory.

Note: This project utilized several custom functions nested within .py scripts within given directories. It is important that the nested script directories and scripts are not moved for they will destroy the PyTorch framework.



## Running Instructions

In the root directory are a the expression_predictor.ipynb Jupyter Notebook and the expression_predictor.py Python script. Both are equivalent to creating facial expression predictions using the model. The images for which to make predictions for are placed in the predictor_images_to_predict directory, the model is executed using either the Jupyter Notebook or the Python script and the predictions are saved in the predictor_images_predicted directory.



## Visualizations

| Model Accuracy                                               |
| ------------------------------------------------------------ |
| ![model_accuracy_viz](.\model_visualizations\model_accuracy_viz.png) |
| **Model Loss**                                               |
| ![model_loss_viz](.\model_visualizations\model_loss_viz.png) |

| VGG19 Confusion Matrix                                       | ResNet18 Confusion Matrix                                    |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![vgg19_model_confusion_matrix](.\model_visualizations\vgg19_model_confusion_matrix.png) | ![resnet18_model_confusion_matrix](.\model_visualizations\resnet18_model_confusion_matrix.png) |



## Predicted Expression Examples

| ![happy_expression_ex](.\predictor_images_predicted\happy_expression_ex.jpg) | ![neutral_expression_ex](.\predictor_images_predicted\neutral_expression_ex.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![sad_expression_ex](.\predictor_images_predicted\sad_expression_ex.jpg) | ![disgust_expression_ex](.\predictor_images_predicted\disgust_expression_ex.jpg) |
| ![angry_expression_ex](.\predictor_images_predicted\angry_expression_ex.jpg) | ![fear_expression_ex](.\predictor_images_predicted\fear_expression_ex.jpg) |



## Conclusions

The CK+ dataset proved to be a very useful dataset to use for creating a model using deep convolutional neural networks. Both VGG19 and ResNet18 led to the creation of very accurate models with low loss rates. The models  were not perfect as both neural networks struggled slightly when distinguishing between some emotions. VGG19 struggled slightly when learning disgust, sadness, and surprise. ResNet18 struggled slightly when learning anger, disgust, fear, sadness, surprise, and contempt. Therefore, when deciding which model to use for making expression predictions, VGG19 was the natural choice. When predicting expressions the model struggled when the image did not have a plain background and it also struggled to distinguish between subtly different emotions when in practice on real world images. Future work would focus on obtaining a larger dataset with more training data that shows a bit more variation in the expressions for each emotion.



## Requirements

h5py == 2.10

Python == 3.8

PyTorch == 1.6

numpy == 1.18.5

matplotlib == 3.2.2

scikit-learn == 0.23.1

opencv-python == 4.4.0.42

Pillow - Python Image Library (PIL) == 7.2.0

Linux or Windows OS Computer

NVIDIA CUDA 11.0 Capable GPU with at least 6GB of memory

## Project Directory

```
|--data
	|--ck+
		|--anger
			|--135_images
		|--contempt
			|--54_images
		|--disgust
			|--177_images
		|--fear
			|--75_images
		|--happy
			|--207_images
		|--sadness
			|--84_images
		|--surprise
			|--249_images
		|--ck_data.h5
	|--model_emojis
		|--7_images
|--model_checkpoints
	|--CK+_Resnet18
		|--emoclass_model.t7
	|--CK+_VGG19
		|--emoclass_model.t7
	|--predictor_weights
		|--deploy.prototxt
		|--weights.caffeemodel
|--model_src
	|--built_models
		|--resnet.py
		|--vgg.py
	|--model_backend
		|--transformers
			|--functional.py
			|--transforms.py
		|--create_confusion_matrix.py
		|--create_train_data.py
		|--progress_bar.py
	|--model_executor.ipynb
	|--model_executor.py
|--model_visualizations
	|--model_accuracy_viz.png
	|--model_loss_viz.png
	|--resnet18_model_confusion_matrix.png
	|--vgg19_model_confusion_matrix.png
|--predictor_images_predicted
	|--6_images
|--predictor_images_to_predict
	|--6_images
|--readme_images
	|--crowd.png
	|--resnet18_architecture.jpg
	|--vgg19_architecture.jpg
|--expression_predictor.ipynp
|--expression_predictor.py
|--expression_recognition_presentation.pdf
|--README.MD
|--requirements.txt
|--LICENSE
|--.gitignore
```



## References

https://www.researchgate.net/figure/Proposed-Modified-ResNet-18-architecture-for-Bangla-HCR-In-the-diagram-conv-stands-for_fig1_323063171

https://www.researchgate.net/figure/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means_fig2_325137356

Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53. 

Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

Simonyan, K., Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. Cornell University. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556) [cs.CV]

He, K., Zhang, X., Ren, S., Sun, J. (2015). Deep Residual Learning for Image Recognition. Cornell University. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) [cs.CV]

oarriaga. (2019, last commit.). Face classification and detection. https://github.com/oarriaga/face_classification

WuJie1010. (2018, last commit.). Facial-Expression-Recognition.Pytorch. https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch