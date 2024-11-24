Machine Learning Using Continuous Data (Potential title - linear regression vs k NN)
---------------------------
We will begin with a recap on linear modelling and then continue onto how we can delve deeper into our data and see how it may change in the future ! We will also introduce `knn.reg` and `arima` to make projections, using algorithms from observed data. This is a continuation from the Machine Learning in R Turorial (see here- https://ourcodingclub.github.io/tutorials/machine-learning/) except we will look at continuous data rather than categorical.

<img align='left' width = '200' height = '150' src='https://github.com/user-attachments/assets/d4a2ae9c-e45c-4988-a839-c9f48980333f' />

### Tutorial Aims

#### <a href="#section1"> 1. Linear modelling</a>

#### <a href="#section2"> 2. The basics of k NN regression</a>


#### <a href="#section3"> 3. ANIMA</a>







### Background Info

Often when analysing data in ecology and environmental science we use modelling to determine the effect explanatory variables have on response variables such as linear regression and generalised linear models. For example, we can determine the extent to which climate change impacts temperatures or precipitation rates in different areas by looking at observed data over time. 

But what if we want to see how this will change in the future? Machine learning can be a useful tool to make projections on future climate trends. In this tutorial we will look at the potentail for linear regression, k NN regression and ARIMA, to project continuous data using algorithms, with the example of annual temperatures across the UK. We will then look at other ways projections can be made using machine learning such as ARIMA models.


You can get all of the resources for this tutorial from <a href="https://github.com/EdDataScienceEES/tutorial-SarahMorrison6504" target="_blank">this GitHub repository</a>. Clone and download the repo as a zip file, then unzip it.

<a name="section1"></a>

## Getting Started
We will be using the dataset uk.long.csv which contains the mean annual temperatures from 1884-2023 in England, Scotland, Wales and Northern Ireland. Be sure to setwd to where you located the unzipped files. We will be using the following packages, so make sure to have them downloaded and loaded onto a new `R Script`:

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
Our output shows that generally there is an increase in 0.009 °C per year (from the estimate column for 'year'. 'year' is the reference country, which in this case is England. The table also tells us that Northern Ireland is on average 0.58 °C, Scotland 2°C and Wales 0.42°C cooler than England.

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

(predict_plot <- (ggplot(temp_predict, aes(x= year, y = ann, colour = Country))+
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
ggplot(future_data, aes(x = year, y = predicted_ann, color = Country)) +  
  geom_line() +
  geom_point() +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci, fill = Country), alpha = 0.2) +
  labs(title = "Projections of Annual Mean Temperature by Country",
       x = "Year",
       y = "Predicted Annual Mean Temperature °C"
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major gridlines
    panel.grid.minor = element_blank(),  # Removes minor gridlines
    axis.line = element_line(color = "black", size = 0.5)  # Adds axes lines
  )
```

<img width = '500' height = '400' src= 'https://github.com/user-attachments/assets/c0e4b8a8-f107-4847-b824-70ffa0c81ef5' />

Hmm, this does show the likely future temperatures however doesn't seem to show similar variability as we see in the observed data from 1884-2023.... maybe we should try machine learning! 

<a name="section2"></a>

## 2.1 Making a kNN regression model

kNN (k -Nearest Neighbours) regression is a form of data mining, that essentially predicts values for new data points by averaging the observations in the same 'neighbourhood'. It doesn't require any assumptions to be made about the data and uses algorithm to make predictions. It works by specifying a k value i.e. the number of neighbouring data points (chosen by the data analyst) to predict the outpout as the average of those points. 

<img width = '600' height = '400' src = 'https://github.com/user-attachments/assets/1f3f0061-3738-4cea-b2ef-27fab2ce1424' /> https://www.youtube.com/watch?v=3lp5CmSwrHI - this video also explains k NN regression well!














For more background on how kNN works, read these articles! - 

- https://www.researchgate.net/publication/358510901_Weather_Prediction_System_Using_KNN_Classification_Algorithm#full-text
- https://bookdown.org/tpinto_home/Regression-and-Classification/k-nearest-neighbours-regression.html






For kNN, there is two stages, the training phase, and the testing phase. For this example we will be using the same dataset, `uk_long` but wil divide the data into thirds (2/3 used for training, 1/3 used for testing). 

## Step 1 - Data Preparation

First, we will ensure that our data is in the correct format. When completing a k NN, r prefers working with numeric values, therefore we add a column numbering each country i.e. England = 1, Scotland = 2...

For k NN and machine learning in general it is best to also scale our data to ensure that all values fall under the same range

```r
# Step 1: Preprocess the data
uk_long$Country <- factor(uk_long$Country)  # make 'Country' a factor, rather than character

# Encode Country as numeric values
uk_long$Country_num <- as.numeric(uk_long$Country)  # add an extra column for Country_num, giving each country a number

# Scale the 'year' and 'Country_num' columns 
uk_long_scaled <- uk_long %>%
  mutate(year = scale(year), Country_num = scale(Country_num))
```
## Step 2 - Split into training and testing

Next, we will split the data into training and testing sets as mentioned earlier. Here we will also `set.seed` before calling `sample`. This ensures that each time we run the code we are taking the same sample ensuring that the output is the same.

```r

# Step 2: Split data into training and testing sets
set.seed(1234)  # generate seed
datasample <- sample(2, nrow(uk_long_scaled), replace = TRUE, prob = c(0.67, 0.33))  # randomly generate our sample with a ratio of 2/3 and 1/3

# Create training and testing sets
temp_training <- uk_long_scaled[datasample == 1, ]  # makes training set from 2/3 of the data 
temp_test <- uk_long_scaled[datasample == 2, ]  # makes testing set from 1/3 of the data
```

## Step 3 - Using the model!


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
## Step 4 - Visualising the model

Now we have our predicted values, however its not very helpful to look at them as a list of numbers...
So lets visualise !!

We can do this by comparing predictions with the actual test data to see how well the model captures the data trends.

```r
# Step 4 - visualising the model

# Combine predictions with actual test data for comparison
combined_data <- data.frame(            # make new dataframe for combined data
  year = temp_test$year,                # uses test data year values for the year column
  ann = temp_test$ann,                  # uses test data annual mean temperature values for ann column
  Country = temp_test$Country,          # uses test country data for country column
  Predicted = knn_predict               # adds column for the model predictions of annual mean temp
)

# Plot predicted vs actual temperatures
(predict_plot <- ggplot(combined_data, aes(x = year)) +  # use ggplot to make plot with combined_data d.f
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




<img width = '500' height = '400' src = 'https://github.com/user-attachments/assets/28aaed4c-f308-4043-afcc-de1ea68cfb0b'/>

<div style="background-color: 'red' ; padding: 10px; border-radius: 5px;">
NOTE: notice how our x- axis for year is from -1, 0 and 1. This is because we scaled our data. We can still see the detailed trends over time though from this graph, but we need to know the time frame that this is from (1884-2023)!


## Step 5 - Error - Root Mean Square Error

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

# 2.2 Making projections from k NN regression

k NN regression can also be useful for making projections like linear models ! We wil use the same example of UK annual mean temperatures to compare the projections made by k NN regression compared to linear regression.

## Step 1



```r
# Define countries and their numeric value

countries <- c(1, 2, 3, 4)  # 1 = England, 2 = Northern Ireland, etc.

# Create an empty list to store projections for each country
all_projections <- list()

# Loop over each country to generate projections
for (country_id in countries) {
  # Create the future data frame for this country
  future_data <- data.frame(Country = rep(country_id, length(future_years)), 
                            Year = scaled_future_years)
  
  # Make predictions for this country using knn.reg
  future_temperatures <- knn.reg(
    train = trainData[, c("Country", "Year")], 
    test = future_data, 
    y = trainData$ann, 
    k = k_value
  )
  
  # Store the projections in the list
  projections <- data.frame(Year = future_years, 
                            Country = country_id, 
                            Predicted_Temperature = future_temperatures$pred)
  
  # Add the projections for this country to the list
  all_projections[[country_id]] <- projections
}

# Combine all country projections into one dataframe
all_projections_df <- bind_rows(all_projections)

# View the projections for all countries
head(all_projections_df)

# Plot the projections with ggplot
(ggplot(all_projections_df, aes(x = Year, y = Predicted_Temperature, color = as.factor(Country))) +
  geom_line(size = 1) +
  labs(title = "Projected Mean Annual Temperature by Country", 
       x = "Year", 
       y = "Predicted Temperature", 
       color = "Country") +  # Customize the legend title
  scale_color_manual(values = c('lightpink', 'lightgreen', 'lightblue','violet'),  # Customize the colors if needed
                     labels = c("England", "Northern Ireland", " Scotland", "Wales")) +  # Set custom country names in the legend
  theme_minimal()+
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.line = element_line(colour = 'black')
    ))







 

## 3. The third section

More text, code and images.

This is the end of the tutorial. Summarise what the student has learned, possibly even with a list of learning outcomes. In this tutorial we learned:

##### - how to generate fake bivariate data
##### - how to create a scatterplot in ggplot2
##### - some of the different plot methods in ggplot2

We can also provide some useful links, include a contact form and a way to send feedback.

For more on `ggplot2`, read the official <a href="https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf" target="_blank">ggplot2 cheatsheet</a>.

Everything below this is footer material - text and links that appears at the end of all of your tutorials.





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
