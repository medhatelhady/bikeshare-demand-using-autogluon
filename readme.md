# Report: Predict Bike Sharing Demand with AutoGluon Solution

# Name: Mahmoud Housam

## Initial Training

### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

I needed to clearly define the label which the oredictor work on to predict and aslo the evaluation metric used which I see is good but there should be a chance to use other metrics.

### What was the top ranked model that performed?

It was `WeightedEnsemble_L3`

## Exploratory data analysis and feature creation

### What did the exploratory analysis find and how did you add additional features?

As suggested, I did a histogram for all of the features in `train.csv`. The histograms shows that there are some columns that should be converted to category type as type include categories not real numbers such as `season, whether, holiday, workingday`. Also, the `datetime` column can be used to exract year, month, day and hour. Both explorations have been executed and affected the results.

### How much better did your model preform after adding additional features and why do you think that is?

After converting the columns mentioned above to category type and extracting `month, day and hour`, the RMSE evaluation metric has dropped from -52.79 to -30.15. The more closed to zero RMSE is, the better it is.

*Note: People might think that returning negative evaluation score is impossible specially with MSE but in RMSE the case is different. Since AutoGloun uses SK-learn to calculate it, the result is always negative due to the inner implementation of calculcation. Check out [this link](https://www.kaggle.com/questions-and-answers/154600). So, it's not a real negative result but it's just the code source that does so*

## Hyper parameter tuning

### How much better did your model preform after trying different hyper parameters?

After doing some feature engineering for the dataset itself by standardizing `temp, atemp, humidity and windspeed` so that the mean is 0 and the standrad deviation is 1 in each column across the two dataframes; train and test and the `time_limit` has been inscreased to around 15 minutes, the performance has slighlty imporved from -30.15 to -29.82. Scaling these features will help the model to not stuck while training at the big numbers and large variance of these features.

### If you were given more time with this dataset, where do you think you would spend more time?

I'll prefer spend more time in the hyperparameters optimization phase where I already did some feature engineering from the dataset. Also, I believe that if accessing the model chosen by AutoGloun is possible, we can modify the hyperparameters of this model to get better results.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.


```python
import pandas as pd
pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": ["eval_metric=root_mean_squared_error", "time_limit=600", "presets=best_quality"],
    "hpo2": ["eval_metric=root_mean_squared_error", "time_limit=600", "presets=best_quality"],
    "hpo3": ["eval_metric=root_mean_squared_error", "time_limit=1000", "presets=best_quality"],
    "score": [1.80, 0.69875, 0.72688]
})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>hpo1</th>
      <th>hpo2</th>
      <th>hpo3</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>initial</td>
      <td>eval_metric=root_mean_squared_error</td>
      <td>eval_metric=root_mean_squared_error</td>
      <td>eval_metric=root_mean_squared_error</td>
      <td>1.80000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>add_features</td>
      <td>time_limit=600</td>
      <td>time_limit=600</td>
      <td>time_limit=1000</td>
      <td>0.69875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hpo</td>
      <td>presets=best_quality</td>
      <td>presets=best_quality</td>
      <td>presets=best_quality</td>
      <td>0.72688</td>
    </tr>
  </tbody>
</table>
</div>



### Create a line plot showing the top model score for the three (or more) training runs during the project.


```python
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

# Read Images 
img = mpimg.imread('model_train_score.png') 

# Output Images 
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x1afe730ac40>




    
![png](output_20_1.png)
    


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.


```python
# Read Images 
img = mpimg.imread('model_test_score.png') 

# Output Images 
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x1afe71f1f10>




    
![png](output_22_1.png)
    


## Summary

From the first plot, we can see the improved evaluation score from -52 to -29 after some more time spent on training and feature engineering process followed.


```python

```
