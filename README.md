### Assignment 14 
## Background

In this Challenge, I have assumed the role of a financial advisor at one of the top five financial advisory firms in the world. Our firm constantly competes with the other major firms to manage and automatically trade assets in a highly dynamic environment. In recent years, our firm has heavily profited by using computer algorithms that can buy and sell faster than human traders.

The speed of these transactions gave our firm a competitive advantage early on. But, people still need to specifically program these systems, which limits their ability to adapt to new data. You’re thus planning to improve the existing algorithmic trading systems and maintain the firm’s competitive advantage in the market. To do so, I will enhance the existing trading signals with machine learning algorithms that can adapt to new data.

## What You're Creating

Combined new algorithmic trading skills with existing skills in financial Python programming and machine learning to create an algorithmic trading bot that learns and adapts to new data and evolving markets.

In a Jupyter notebook, you’ll do the following:

* Implement an algorithmic trading strategy that uses machine learning to automate the trade decisions.

* Adjust the input parameters to optimise the trading algorithm.

* Train a new machine learning model and compare its performance to that of a baseline model 
Establish a Baseline Performance
In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

## Instructions

Use the starter code file to complete the steps that the instructions outline. The steps for this Challenge are divided into the following sections:

* Establish a Baseline Performance

* Tune the Baseline Trading Algorithm

* Evaluate a New Machine Learning Classifier

* Create an Evaluation Report

### Establish a Baseline Performance

In this section, you’ll run the provided starter code to establish a baseline performance for the trading algorithm. To do so, complete the following steps.

Open the Jupyter notebook. Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four.

1. Import the OHLCV dataset into a Pandas DataFrame.

2. Generate trading signals using short- and long-window SMA values.

3. Split the data into training and testing datasets.

4. Use the `SVC` classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.

5. Review the classification report associated with the `SVC` model predictions.

6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.

7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. Save a PNG image of this plot. This will serve as a baseline against which to compare the effects of tuning the trading algorithm.

8. Write your conclusions about the performance of the baseline trading algorithm in the `README.md` file that’s associated with your GitHub repository. Support your findings by using the PNG image that you saved in the previous step.

### Tune the Baseline Trading Algorithm

In this section, you’ll tune, or adjust, the model’s input features to find the parameters that result in the best trading outcomes. (You’ll choose the best by comparing the cumulative products of the strategy returns.) To do so, complete the following steps:

1. Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing the training window?


2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results in your `README.md` file. Answer the following question: What impact resulted from increasing or decreasing either or both of the SMA windows?

Answer: when I changed the window of the sma prameters to 7 for short and 50 for long, we achieved a recall of 1. Which improved by adittional 4%. 


precision    recall  f1-score   support

         0.0       0.00      0.00      0.00        21
         1.0       0.64      1.00      0.78        38

    accuracy                           0.64        59
   macro avg       0.32      0.50      0.39        59
weighted avg       0.41      0.64      0.50        59



3. Choose the set of parameters that best improved the trading algorithm returns. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns, and document your conclusion in your `README.md` file.

### Evaluate a New Machine Learning Classifier

In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model. To do so, complete the following steps:

1. Import a new classifier, such as `AdaBoost`, `DecisionTreeClassifier`, or `LogisticRegression`. (For the full list of classifiers, refer to the [Supervised learning page](https://scikit-learn.org/stable/supervised_learning.html) in the scikit-learn documentation.)

2. Using the original training data as the baseline model, fit another model with the new classifier.

3. Backtest the new model to evaluate its performance. Save a PNG image of the cumulative product of the actual returns vs. the strategy returns for this updated trading algorithm, and write your conclusions in your `README.md` file. Answer the following questions: Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?

### Create an Evaluation Report

In the previous sections, you updated your `README.md` file with your conclusions. To accomplish this section, you need to add a summary evaluation report at the end of the `README.md` file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.

our first orginal model with short window of 4 days and long window of 100 days,and the model produced a recall value of 56%. Thus making the chosen SMA pramateres a decent model. 

Original Actual retrun are posted below. 

    	                  close	Actual Returns
date		
2015-01-21 11:00:00	23.98	0.006295
2015-01-22 15:00:00	24.42	0.018349
2015-01-22 15:15:00	24.44	0.000819
2015-01-22 15:30:00	24.46	0.000818
2015-01-26 12:30:00	24.33   -0.005315
                           
                           close	Actual Returns
date		
2021-01-22 09:30:00	33.27   -0.006866
2021-01-22 11:30:00	33.35	0.002405
2021-01-22 13:45:00	33.42	0.002099
2021-01-22 14:30:00	33.47	0.001496
2021-01-22 15:45:00	33.44   -0.00089


7 and 50 SMA model posted below

	                  sma_fast	         sma_slow
date		
2015-02-23 15:30:00	24.477143	24.2200
2015-02-23 15:45:00	24.472857	24.2302
2015-02-24 10:45:00	24.507143	24.2362
2015-02-24 11:00:00	24.541429	24.2422
2015-02-24 12:15:00	24.570000	24.2470
...	...	...
2021-01-22 09:30:00	33.065714	31.2232
2021-01-22 11:30:00	33.165714	31.2848
2021-01-22 13:45:00	33.247143	31.3480
2021-01-22 14:30:00	33.292857	31.4132
2021-01-22 15:45:00	33.345714	31.4768


    
    precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092



when I sliced the data to input new pramaters of short 7 days and 50 days, we where able to achieve a 100% percision on profitable and 0 on unporofitable, and our recal increased to 64% which can make this model a stronger model. 


     precision    recall  f1-score   support

         0.0       0.00      0.00      0.00        21
         1.0       0.64      1.00      0.78        38

    accuracy                           0.64        59
   macro avg       0.32      0.50      0.39        59
weighted avg       0.41      0.64      0.50        59




unfortunatley when I chose logistic Regression, the didnt change much from the original predictions. 

     precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092

