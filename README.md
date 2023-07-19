# Model Training and Inference

This repository contains code for training and inference using a deep learning model.


## Introduction
This project aims to train and deploy a deep learning model for a specific task. The model is trained on a labeled dataset and later used to predict labels for new unseen data.

## Installation
- Clone this repository:

```bash
git clone git@github.com:yusufbrima/FlowersNet.git
```

- Install the required packages:

```bash
cd your_repository
pip install -r requirements.txt
```

## Data Preparation
- Download the dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html) and extract it in the `data` directory.

### Data Samples 
![Data Samples](Figures/plot.png)

## Model Training
- Train the model using the following command:

```bash
python main.py
```

## Model Inference
- Use the model to predict labels for new data using the following command:

```bash
python predict.py
```

# Model Performance

This table shows the performance metrics of the model on the training, validation, and test datasets.

| Dataset   | Accuracy | Loss    |
|-----------|----------|---------|
| Training  | 0.95     | 0.84    |
| Validation| 0.96     | 0.84    |
| Test      | 0.88     | 0.91    |


## Sample Output

![Sample Output](Figures/prediction_plot.png)

## License
This project is licensed under the terms of the [MIT license]().
