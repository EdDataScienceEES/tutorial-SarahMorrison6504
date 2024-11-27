## This script is how the randomly generated climate data was generated

set.seed(123)  # set seed so that generated data is the same each time
n_countries <- 100  # Increase the number of rows
climate_data <- data.table(
  Temperature_Change = rnorm(n_countries, mean = 1, sd = 0.5),  # Random temperature changes (°C)
  Precipitation_Change = rnorm(n_countries, mean = 0, sd = 30))  # Random precipitation changes (mm)

write.csv(climate_data, 'climate_data.csv')
