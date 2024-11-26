# Linear regression vs k NN) 
<img align='right' width = '200' height = '150' src='https://github.com/user-attachments/assets/d4a2ae9c-e45c-4988-a839-c9f48980333f' />

We will begin with a recap on linear modelling and then continue onto how we can delve deeper into how we can use modelling for future predictions! We will also introduce `knn.reg` and `arima` to make projections, using algorithms from observed data. This is a continuation from the Machine Learning in R Turorial ([see here](https://ourcodingclub.github.io/tutorials/machine-learning/)) except we will look at continuous data rather than categorical.



### Tutorial Aims
1. Recap linear regression and modelling from [Intro to model design](https://ourcodingclub.github.io/tutorials/model-design/)
2. Develop and understanding of how linear regression can be used to make future projections of data
3. Introduce `knn.reg` and how it can be used to model data
4. Learn how `knn.reg` can be used to make data projections
5. Recognise when it is appropriate to use each model.

### Steps

#### <a href="#section1"> 1. Linear regression</a>

#### <a href="#section2"> 2. The basics of k NN regression</a>

#### <a href="Comparing k NN regression and Linear Regression for making future projections of data"> 3. Comparing Linear regression and k NN regression </a>







### Introduction

Regression is used to estimate the relation between a response variable and at least one predictor variable. This is often used in data analysis in ecology and environmental science. For example, we can determine the extent to which climate change impacts temperatures or precipitation rates in different areas by looking at observed data over time. 

But what if we want to see how this will change in the future? We can use linear regression, but what if we want more detailed projections? Or our data is not normally distruibuted? Machine learning can be a useful tool for this and  we will look at the potentail for one method, k NN regression, to project continuous data using algorithms, with the example of annual temperatures across the UK.


You can get all of the resources for this tutorial from <a href="https://github.com/EdDataScienceEES/tutorial-SarahMorrison6504" target="_blank">this GitHub repository</a>. Clone and download the repo as a zip file, then unzip it.

<a name="section1"></a>

## 1. Linear Regression 
### Getting Started
We will be using the dataset `uk_long.csv` which contains the mean annual temperatures from 1884-2023 in England, Scotland, Wales and Northern Ireland. This dataset was created by compiling data from the [Met Office](https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series). 
(See how this was compiled in the [data script](https://github.com/EdDataScienceEES/tutorial-SarahMorrison6504/tree/master/data)!)


Be sure to setwd to where you located the unzipped files. We will be using the following packages, so make sure to have them downloaded and loaded onto a new `R Script`:

```r
# Set the working directory
setwd("your_filepath")

# Load packages
library(caret)
library(dplyr)
library(tidymodels)
library(ggplot2)
library(tidyr)
library(gmodels)
library(randomForest)
library(FNN)
library(ggeffects)
```
First, lets inspect our data:

```r
str(uk_long)  # view structure of data
head(uk_long)  # get summary of first 6 rows
```


We can see that we have a dataset with 3 variables, 'year', 'ann' and 'Country'. Here 'ann' is annual mean temperature. 

### Lets visualise our data
(TIP: use () in the command to automatically see the plot!)

```r
(scatter <- ggplot(uk_long, aes(x = year, y = ann, colour = Country))+  # use ggplot to make scatter plot
    geom_line(size = 1, alpha = 0.6)+  # adjust size of line
    theme_classic()+
    theme(legend.position = 'right'))  # move legend to the right
```

<img align = 'centre' width = '500' height = '300' src= 'https://github.com/user-attachments/assets/54466297-63ec-4c01-bf21-877bc0766364' />


We could model this data to determine the extent of the trends in our data.
First find the distribution:

```r
hist(uk_long$ann)  # plot histogram to see distribution of our data
```
<img align = 'centre' width = '500' height = '400' src= 'https://github.com/user-attachments/assets/a41073af-e873-45ee-9953-1f31d1ec5f10' />


As our trends are normally distributed we could use a linear regression to show the changes in temperature over time in each country. 

```r
annual_lm <- lm(ann ~ year + Country, data = uk_long)  # model considering country as a fixed effect

summary(annual_lm)  # summary table of model

```
Output:
```r
Coefficients:
                         Estimate Std. Error t value Pr(>|t|)    
(Intercept)             -8.439224   0.995177  -8.480  < 2e-16 ***
year                     0.009033   0.000509  17.747  < 2e-16 ***
CountryNorthern Ireland -0.584786   0.058182 -10.051  < 2e-16 ***
CountryScotland         -2.055071   0.058182 -35.322  < 2e-16 ***
CountryWales            -0.419857   0.058182  -7.216 1.76e-12 ***
```
Our output shows that generally there is an increase in 0.009 °C per year (from the estimate column for 'year'. 'year' is the reference country, which in this case is England. The table also tells us that Northern Ireland is on average 0.58 °C, Scotland 2°C and Wales 0.42°C cooler than England. The standard error is also small suggesting that the fit is good.

We can check the model is appropriate by checking the residuals

```r
plot(annual_lm)
```
Which gives us:


<img width = '400' height = '400' src= 'https://github.com/user-attachments/assets/82e9a77b-5403-48de-919a-f42ce9b62f6a' />

So our model has a good fit ! Which shows that the estimates on increases per year are pretty accurate:)
We can then plot our model predictions against the actual data to visualise the accuracy of the model better!

```r

temp_predict <- uk_long  # change the name of uk_long dataframe to include predictions
temp_predict$Predicted <- predict(annual_lm, newdata = temp_predict, type = 'response')  # add column for model predictions

```
Now we have a new dataframe called temp_predict with the model predictions as an extra column. We can then use this to visualise the model against observed data:
```r

(lmpredict_plot <- (ggplot(temp_predict, aes(x= year, y = ann, colour = Country))+
  geom_line(aes(y= ann, colour = Country), size =1, linetype = 'solid')+
  geom_line(aes(y= Predicted, colour = Country), size = 1, linetype = 'dashed')+
  labs(title = 'Predicted vs Actual Temperatures', x = 'Actual', y ='Predicted')))

```
Which gives us: 

<img width = '500' height = '400' src = 'https://github.com/user-attachments/assets/e1b4ddef-ec99-408d-b7a2-5224ad6cb968' />

Modelling can be useful for looking at trends in data and predicting values within the time frame. For example if we wanted to predict what the temperature would be on a specific month within the years of 1884-2023, we could make an estimate from our linear model! We can also use our linear model to make projections of future temperatures!

First we make a new data frame with columns for Country and future years (here we will look at 2025-2030)

```r
# Define future years
future_years <- 2025:2030

# Define countries
countries <- c("England", "Scotland", "Wales", "Northern Ireland")

# Create a data frame with all combinations of future years and countries using expand.grid
future_data <- expand.grid(year = future_years, Country = countries)

```
Next we will use our model to make future predictions

```r
future_data$predicted_ann <- predict(annual_lm, newdata = future_data)  # add column called predicted_ann for our model predictions
```

We can also add confidence intervals (ci) to quantify the imprecision associated with our estimates

```r
predictions_with_ci <- predict(annual_lm, newdata = future_data, interval = 'confidence')  # add ci for predictions

future_data$lower_ci <- predictions_with_ci[, "lwr"]  # add column for lower ci
future_data$upper_ci <- predictions_with_ci[, "upr"]  # add column for upper ci

```

Now we can plot our future predictions!

```r
(linear_future_predict <- ggplot(future_data, aes(x = year, y = predicted_ann, color = Country)) +  
  geom_line() +
  geom_point() +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci, fill = Country), alpha = 0.2) +
  labs(title = "Future Temperature Predictions (Linear Regression)",
       x = "Year",
       y = "Predicted Mean Temperature °C"
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major gridlines
    panel.grid.minor = element_blank(),  # Removes minor gridlines
    axis.line = element_line(color = "black", size = 0.5)  # Adds axes lines
  )))
```

<img width = '500' height = '400' src= 'https://github.com/user-attachments/assets/c0e4b8a8-f107-4847-b824-70ffa0c81ef5' />

Hmm, this does show the likely future temperatures however doesn't seem to show similar variability as we see in the observed data from 1884-2023.... maybe we should try a form of machine learning, k NN regression! 

<a name="section2"></a>

## 2. kNN Regression Model
### 2.1 Creating a kNN regression model
kNN (k -Nearest Neighbours) regression is a form of data mining, that essentially predicts values for new data points by averaging the observations in the same 'neighbourhood'. It doesn't require any assumptions to be made about the data and uses algorithm to make predictions. It works by specifying a k value i.e. the number of neighbouring data points (chosen by the data analyst) to predict the outpout as the average of those points. 

<img width = '600' height = '400' src = 'https://github.com/user-attachments/assets/1f3f0061-3738-4cea-b2ef-27fab2ce1424' />

[this video also explains k NN regression well!](https://www.youtube.com/watch?v=3lp5CmSwrHI)















For more background on how kNN works, read these articles! - 
- [Weather Prediction System Using KNN Classification Algorithm](https://www.researchgate.net/publication/358510901_Weather_Prediction_System_Using_KNN_Classification_Algorithm#full-text)
- [BioStatistics- K-nearest neightbours regression](https://bookdown.org/tpinto_home/Regression-and-Classification/k-nearest-neighbours-regression.html)






For kNN, there is two stages, the training phase, and the testing phase. For this example we will be using the same dataset, `uk_long` but wil divide the data into thirds (2/3 used for training, 1/3 used for testing). 

### Step 1 - Data Preparation

First, we will ensure that our data is in the correct format. When completing a k NN, r prefers working with numeric values, therefore we add a column numbering each country i.e. England = 1, Scotland = 2...

For k NN and machine learning in general it is best to also scale our data to ensure that all values fall under the same range

```r
# Step 1: Preprocess the data
uk_long$Country <- factor(uk_long$Country)  # make 'Country' a factor, rather than character

# Encode Country as numeric values
uk_long$Country_num <- as.numeric(uk_long$Country)  # add an extra column for Country_num, giving each country a number

# Scale the 'year' and 'Country_num' columns
year_centre <- mean(uk_long$year, na.rm = TRUE)  # uses the mean of 'year' in the original data
year_scale <- sd(uk_long$year, na.rm = TRUE)     # uses the standard deviation of 'year' in the original data

uk_long_scaled <- uk_long %>%
 mutate(year = scale(year),
 Country_num = scale(Country_num))
```
### Step 2 - Split into training and testing

Next, we will split the data into training and testing sets as mentioned earlier. Here we will also `set.seed` before calling `sample`. This ensures that each time we run the code we are taking the same sample ensuring that the output is the same.

```r

# Step 2: Split data into training and testing sets
set.seed(1234)  # generate seed
datasample <- sample(2, nrow(uk_long_scaled), replace = TRUE, prob = c(0.67, 0.33))  # randomly generate our sample with a ratio of 2/3 and 1/3

# Create training and testing sets
temp_training <- uk_long_scaled[datasample == 1, ]  # makes training set from 2/3 of the data 
temp_test <- uk_long_scaled[datasample == 2, ]  # makes testing set from 1/3 of the data
```

### Step 3 - Using the model!


Next we can perform the model. When using `knn.reg()` we have to assign a value to `k` representing he number of neighbours considered in the analysis. Here we will use `k <- 5`, which tells the model that we want it to use 5 neighbours when making predictions. 
We will then provide the response variable `ann` to tell the model what we want it to make predictions of. We will then use the training and testing sets from earlier to make these predictions. 

```r
# Step 3: Using knn.reg() for predictions
k <- 5  # number of neighbours
train_response <- temp_training$ann  # here we are telling the model what we want the response variable to be, 'ann' (mean annual temperature)

knn_predict <- knn.reg(
  train = temp_training[, c("year", "Country_num")],  # predictors in the data for training
  test = temp_test[, c("year", "Country_num")],      # predictors in the data for testing
  y = train_response,                                # response variable
  k = k                                              # k value that we set earlier
)$pred                                               # extracts predictions from the model

print(knn_predict)  # print the model predicted mean annual temperatures

```
### Step 4 - Visualising the model

Now we have our predicted values, however its not very helpful to look at them as a list of numbers...
So lets visualise !!

We can do this by comparing predictions with the actual test data to see how well the model captures the data trends. Remember we have to unscale our variables we scaled earlier so that we can compare to the actual values.

```r
# Step 4 - visualising the model
temp_test$unscaled_year <- temp_test$year * year_scale + year_center  # unscale values

# Combine predictions with actual test data for comparison
combined_data <- data.frame( # make new dataframe for combined data
temp_test$unscaled_year <- temp_test$year * year_scale + year_center,  # unscale year  
  year = temp_test$year,                # uses test data year values for the year column
  ann = temp_test$ann,                  # uses test data annual mean temperature values for ann column
  Country = temp_test$Country,          # uses test country data for country column
  Predicted = knn_predict               # adds column for the model predictions of annual mean temp
)

# Plot predicted vs actual temperatures
(predict_plot <- ggplot(combined_data, aes(x = unscaled_year)) +  # use ggplot to make plot with combined_data d.f
  geom_line(aes(y = ann, colour = Country), size = 1, linetype = 'solid') +  # make solid line for test data values
  geom_line(aes(y = Predicted, colour = Country), size = 1, linetype = 'dashed') +  # make dashed line for model predictions
  labs(title = 'Predicted vs Actual Temperatures (k-NN)', x = 'Year', y = 'Temperature') +  # add title
  theme_minimal()+
theme(
panel.grid = element_blank(),  # removes gridlines
axis.line = element_line(colour = 'black')  # adds axes lines
))
```
OUTPUT




<img width = '500' height = '400' src = 'https://github.com/user-attachments/assets/84d826c2-7593-426c-a7c3-c5087988ed28'/>

In our plot the dashed lines are the model predictions, and we can see that there is more variation than the results of the linear regression, with it matching observed data more. However there is still error associated.






### Step 5 - Error - Root Mean Square Error

When modelling, it is often useful to find an error value to take into account the uncertainties with the model. For a k NN regression, a RMSE (root mean square value) is often appropriate. The RMSE essential shows the prediction error of the model. A lower RMSE indicated that the model has high predictive accuracy. Altering the `k` value we assigned as 5 earlier is a way to lower the RMSE. (For more info on RMSE follow this link ! - https://www.sciencedirect.com/topics/engineering/root-mean-square-error)

<img width = '600' height = '300' src = 'https://github.com/user-attachments/assets/518b328b-4b8d-489c-ba4f-c1cf06f57b4e' />



</div>

```r
rmse <- sqrt(mean((combined_data$ann - combined_data$Predicted)^2, na.rm = TRUE))  # formula to calculate RMSE

print(paste('RMSE', rmse))  # prints the RMSE value in the console

```
OUTPUT

```r
"RMSE 0.502558000113679"
```

This tells us that for each prediction made by the model, it is inaccurate by on average 0.5 °C from the true (observed values). 0.5°C is a fairly good RMSE especially as we are only taking into account 2 varaibles (`year` and `ann`).
Taking into account seasonal changes for example, may increase the accuracy of our models predictions, but for simplicity we will leave the model as is!

### 2.2 Making projections from k NN regression

k NN regression can also be useful for making projections like linear models ! We wil use the same example of UK annual mean temperatures to compare the projections made by k NN regression compared to linear regression.

### Step 1- Setting up variables

First we need to set up the variables we want to make predictions on. We can do this by making new objects,  `future years` for the range of years we want to include and `countries` to specify what countries we want to make projections of. We can then make a dataframe from these objects. Additionally, we want Country to be a factor, so we must add a new numeric column for country, ( i.e. 1, 2 , 3, 4 each corresponding to a country) for analysis.

```r
# 1. Setting up variables
future_years <- seq(2024, 2030) # set the range of future years we want to project
countries <- unique(uk_long$Country)  # unique() finds the different countries in uk_long and makes them an object, 'countries'

future_data <- expand.grid(year = future_years,
Country = countries)  # makes a data frame for future years and countries

future_data$Country_num <- as.numeric(factor(future_data$Country))  # makes levels of the factor variable country, as a new numeric column 'Country_num'
```

### Step 2 - Scale the data

Like when we used k NN regression for the observed data, we need to scale our variables `Country_num` and `year`. 
```
# 2. Scale the new data

future_data_scaled <- future_data %>%
  mutate(year = scale(year), Country_num = scale(Country_num))  # Scale 'year' and 'Country_num' like the original data
```

### Step 3 - Using our previous k-NN model make projections
We will use `temp_training` data from our k NN regression model previously to train the model. the `future_data_scaled` data will be what the predictions will be made for and `train_response` (mean annual temperature) will be the variable that the model is trying to predict. `k` remains the same as previously.
```
# 3. Using the model
knn_predictions <- knn.reg(
  train = temp_training[, c("year", "Country_num")],  # using our training data's features
  test = future_data_scaled[, c("year", "Country_num")],  # using our scaled future data's features
  y = train_response,  # using the actual 'ann' values from the training set
  k = k
)$pred

```

### Step 4 - Visualising model results

First we need to combine our predictions with the `future_data` dataframe and add them as a new column. Then we need to make sure that `Country` in `future_data` is a factor and matches the same level as they are in the training data so that the countries are represented correctly. Then we can plot our resuts !

```r
# 4. Visualising model

future_data$Predicted_ann <- knn_predictions  # combine our predictions with the future_data dataframe
future_data$Country <- factor(future_data$Country, levels = countries)  # Ensure factor levels match



(future_predict_plot <- ggplot(future_data, aes(x = year, y = Predicted_ann, colour = Country)) +
  geom_line(size = 1) +
  labs(title = 'Future Temperature Predictions (k-NN)',
 x = 'Year',
 y = 'Predicted Mean Temperature °C') +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(colour ='black')
  )))

print(future_data)  # we can also see the predictions made in the console
```
OUTPUT

<img width = '500' height = '400' src = 'https://github.com/user-attachments/assets/08094337-3eca-4e5a-8fc9-1205b0e569d4' />


Okay, lets compare this to the results of the linear regression predictions...

```r
panel_plot <- future_predict_plot + linear_future_predict
print(panel_plot)

```

OUTPUT
<img width = '600' height = '400' src = 'https://github.com/user-attachments/assets/2d2f9dca-3c69-44d1-bef2-dafe6fb68726' />



## 3. Comparing k NN regression and Linear Regression for making future projections of data
From this we can see that the k NN has a lot more variability throughout the years, which we would expect when we compare this to our plot of the observed data, which is likely due to the model not having any assumptions that the data behaves in a linear manner. However, we also must consider that the main basis that k NN works on algorithm and does not show the actual effect that each predictor has as a value. Additionally we should keep in mind that the results of the k NN will change depending on the value of `k` assigned. If our dataset had multiple climate explanatory variables, for example seasonal temperatures and humidity, we would find a k NN regression to make more accurate projections than linear regression.

Linear regression still captures the general future trends however, which for the likes of temperature changes, is helpful and easy to communicate to audiences. There is also a summary table associated which explains where the predictions come from which is helpful to
back up why these projections are expected. It also must be considered that linear regression can only be used if data matches the assumptions of the model (data is normally distributed, data points are independant, variance is similar between groups). 

### Summary table of the two types of model

Here are some things to consider if you are thinking of using a linear regression or k NN regression for data analysis:

<img width = '800' height = '300' src = 'https://github.com/user-attachments/assets/9e697535-365b-4272-a808-e3fe0e99ebcc' />



## Questions ?

#### 1. So.. using this summary of these two models, which do you think is most appropriate for modelling the observed data`uk_long`  we have looked at, and for making predictions during the time of observed data?
<details>
	<summary>Click to see the answer</summary>
	Though both show the same general trend of the observed data and linear regression has a small amount of error associated. k NN highlights the variability that occurs in data, which can be useful when making accurate predictions within the observed data time frame. By adding more variables into a k NN regression such as seasonal temperatures, we could have even more accurate estimates.
</details>



#### 2. What about the projected data?
<details>
	<summary>Click to see the answer</summary>
	Considering that we have data that is linear and doesn't include too many explanatory variables, linear regression may be more appropriate and can also communicate the average increase in temperature per year.
</details>


---------------------------
 So what have we learned?

 ## Learning outcomes
 1. You should now be more confident using linear regression and how it can be used to make data projections
 2. You should be familiar with `knn.reg` as a method of modelling data
 3. You should be able to distinguish when it is appropriate to use each kind of model
 4. You should be able to identify the main pros and cons of each model type





<hr>
<hr>

#### Check out our <a href="https://ourcodingclub.github.io/links/" target="_blank">Useful links</a> page where you can find loads of guides and cheatsheets.

#### If you have any questions about completing this tutorial, please contact us on ourcodingclub@gmail.com

#### <a href="INSERT_SURVEY_LINK" target="_blank">We would love to hear your feedback on the tutorial, whether you did it in the classroom or online!</a>

<ul class="social-icons">
	<li>
		<h3>
			<a href="https://twitter.com/our_codingclub" target="_blank">&nbsp;Follow our coding adventures on Twitter! <i class="fa fa-twitter"></i></a>
		</h3>
	</li>
</ul>

### &nbsp;&nbsp;Subscribe to our mailing list:
<div class="container">
	<div class="block">
        <!-- subscribe form start -->
		<div class="form-group">
			<form action="https://getsimpleform.com/messages?form_api_token=de1ba2f2f947822946fb6e835437ec78" method="post">
			<div class="form-group">
				<input type='text' class="form-control" name='Email' placeholder="Email" required/>
			</div>
			<div>
                        	<button class="btn btn-default" type='submit'>Subscribe</button>
                    	</div>
                	</form>
		</div>
	</div>
</div>
