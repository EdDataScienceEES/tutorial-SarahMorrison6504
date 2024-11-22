Machine Learning Using Continuous Data (Potential title - linear regression vs k NN)
---------------------------
We will begin with a recap on linear modelling and then continue onto how we can delve deeper into our data and see how it may change in the future ! We are using `knn.reg` to make projections, using algorithms from observed data. This is a continuation from the Machine Learning in R Turorial (see here- https://ourcodingclub.github.io/tutorials/machine-learning/) except we will look at continuous data rather than categorical. Finally, we will map out these projections and see how these can be visualised for communicating results.

<img align='left' width = '200' height = '150' src='https://github.com/user-attachments/assets/d4a2ae9c-e45c-4988-a839-c9f48980333f' />

### Tutorial Aims

#### <a href="#section1"> 1. Recap on linear modelling</a>

#### <a href="#section2"> 2. Making data projections using k NN regression</a>

#### <a href="#section3"> 3. Mapping trends</a>







### Background Info

Often when analysing data in ecology and environmental science we use modelling to determine the effect explanatory variables have on response variables such as linear regression and generalised linear models. For example, we can determine the extent to which climate change impacts temperatures or precipitation rates in different areas by looking at observed data over time. 
But what if we want to see how this will change in the future? Machine learning can be a useful tool to make projections on future climate trends. In this tutorial we will use a basic k NN regression to demonstrate the potential for using Rstudio to project continuous data using algorithms, with the example of annual temperatures across the UK. We will then look at how these projections can be communicated.

NOTE: This tutorial aims to introduce the ideas of k NN regression however there is always a level of uncertainty when making projections.

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

<img width = '500' height = '300' src= 'https://github.com/user-attachments/assets/54466297-63ec-4c01-bf21-877bc0766364' />


We could model this data to determine the extent of the trends in our data.
First find the distribution:

```r
hist(uk_long$ann)  # plot histogram to see distribution of our data
```
<img width = '500' height = '400' src= 'https://github.com/user-attachments/assets/a41073af-e873-45ee-9953-1f31d1ec5f10' />


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

## 2. Making data projections using kNN regression

kNN (k -Nearest Neighbours) regression is a form of data mining, that essentially predicts values for new data points by averaging the observations in the same 'neighbourhood'. It doesn't require any assumptions to be made about the data and uses algorithm to make predictions. 

For kNN, there is two stages, the training phase, and the testing phase. For this example we will be using the same dataset, `uk_long` but wil divide the data into thirds (2/3 used for training, 1/3 used for testing). To assess the accuracy of our kNN regression we will find a RMSE value.




For more background on how kNN works, read these articles! - 

>https://www.researchgate.net/publication/358510901_Weather_Prediction_System_Using_KNN_Classification_Algorithm#full-text

>https://bookdown.org/tpinto_home/Regression-and-Classification/k-nearest-neighbours-regression.html

First, we will divide our data into training and testing samples.

```r
datasample <- sample(2, nrow(uk_long), replace = TRUE, prob = c(0.67, 0.33))  # dividing data into training and testing samples.

temp_training <- uk_long[datasample == 1,] %>%
  select(year, Country, ann) %>%
  mutate(Country = as.numeric(factor(Country)))

temp_test <- uk_long[datasample == 2,] %>%
  select(year, Country, ann) %>%
  mutate(Country = as.numeric(factor(Country)))  



You can add more text and code, e.g.

```r
# Create fake data
x_dat <- rnorm(n = 100, mean = 5, sd = 2)  # x data
y_dat <- rnorm(n = 100, mean = 10, sd = 0.2)  # y data
xy <- data.frame(x_dat, y_dat)  # combine into data frame
```

Here you can add some more text if you wish.

```r
xy_fil <- xy %>%  # Create object with the contents of `xy`
	filter(x_dat < 7.5)  # Keep rows where `x_dat` is less than 7.5
```

And finally, plot the data:

```r
ggplot(data = xy_fil, aes(x = x_dat, y = y_dat)) +  # Select the data to use
	geom_point() +  # Draw scatter points
	geom_smooth(method = "loess")  # Draw a loess curve
```

At this point it would be a good idea to include an image of what the plot is meant to look like so students can check they've done it right. Replace `IMAGE_NAME.png` with your own image file:

<center> <img src="{{ site.baseurl }}/IMAGE_NAME.png" alt="Img" style="width: 800px;"/> </center>

<a name="section1"></a>

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
