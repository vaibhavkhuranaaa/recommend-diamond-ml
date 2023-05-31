#Clear the environment
rm(list=ls())

#Libraries
library(dplyr)
library(ggplot2)

#PART-1
#List the files available in the working directory
list.files()

#(1A) Load the converted file (excel to csv) and analyse the structure of the variables present.
diamonds <- read.csv("IDS472HW4diamond.csv")

#check if data is not misisng
test_row <- diamonds[7000,]
print(test_row)

#delete the rows with missing Price value for appropriate data preparation
diamonds <- diamonds[!diamonds$Price == "",]
str(diamonds)

#(1B) Exploratory data analysis.
# remove the dollar sign from the Price column.
diamonds$Price <- gsub("[\\$,]", "", diamonds$Price)
# Convert from character to numeric.
diamonds$Price <- as.numeric(diamonds$Price)


#find outliers in price
summary(diamonds$Price)
sd(diamonds$Price)
hist(diamonds$Price)
hist(log(diamonds$Price))
#handle outliers
diamonds$Price <- log(diamonds$Price)
View(diamonds)

# create a scatter plot of carat weight against price
plot(diamonds$Carat.Weight, diamonds$price, main="Scatter plot of Carat Weight vs. Price", xlab="Carat Weight", ylab="Price")

# create a box plot of price against cut
boxplot(Price ~ Cut, data=diamonds, main="Box Plot of Price vs. Cut", xlab="Cut", ylab="Price")

# create a box plot of price against clarity
boxplot(Price ~ Clarity, data=diamonds, main="Box Plot of Price vs. Clarity", xlab="Clarity", ylab="Price")

#(1C) Splitiing the data set into 70% and %30
#Convert all characters into factors for easier analysis
diamonds[sapply(diamonds, is.character)] <- lapply(diamonds[sapply(diamonds, is.character)], as.factor)
summary(diamonds)

# split the dataset
set.seed(123)
train_index <- sample(2, 6000, prob = c(0.7, 0.3), replace = T)

train <- diamonds[train_index == 1, ]
test <- diamonds[train_index == 2, ]

#(1D) Train a linear regression model
linreg_model <- lm(formula = Price ~ .,
                   data = train)

#(1D) AND (1E) 
# Check number of significant variables and R^2
summary(linreg_model)
'Number of variables significant in the model is 19'
'R^2 value of the model is 96%'

#(1F) MAPE
# MAPE on training data
linreg_trainpreds <- (linreg_model$fitted.values)
err_train <- exp(linreg_trainpreds) - exp(train$Price)
abserr_train <- abs(err_train)
percabserr_train <- abserr_train / exp(train$Price)
mape_train <- mean(percabserr_train)
mape_train

#(1G) Over priced diamond according to the model

over_priced <- train %>% mutate(pred_error =exp(linreg_trainpreds) - exp(train$Price)) %>% filter(pred_error == max(pred_error))
print(over_priced)

#(1H) MAPE on testing data
linreg_testpreds <- predict(linreg_model, test)
err_test <- exp(linreg_testpreds) - exp(test$Price)
abserr_test <- abs(err_test)
percabserr_test <- abserr_test / exp(test$Price)
mape_test <- mean(percabserr_test)
mape_test

#(1I) # assuming the training dataset is called "diamonds_train"
recommended_diamond <- train %>% mutate(predicted_price = (linreg_model$fitted.values)) %>%
  filter(predicted_price <= log(12000)) %>%
  slice_max(Carat.Weight)

# output the details of the recommended diamond
print(recommended_diamond)




#PART-2
#preparing the data for model building
library(fastDummies)
diamonds2 <- fastDummies::dummy_cols(diamonds)
diamonds2 <- diamonds2 %>% select(-Cut, -Color, -Clarity, -Polish, -Symmetry, -Report, -ID)

# You need to scale the data
mins <- apply(diamonds2, 2, min)
maxs <- apply(diamonds2, 2, max)
diamonds3 <- scale(diamonds2, mins, maxs-mins) # (x[j,i] - min[,i]) / (max[,i]-min[,i])

train_nn <- diamonds3[train_index == 1, ]
test_nn <- diamonds3[train_index == 2, ]

#Convert arrays to datasets
train_nn <- as.data.frame(train_nn)
test_nn <- as.data.frame(test_nn)

library(nnet)
#(2A) Building neaural network
# Define a function to train and test a neural network model
train_and_test <- function(num_neurons) {
  # Train the model
  nn_model <- nnet(Price ~ ., data = train_nn, size = num_neurons, linout = F, decay=0.01, maxit=100)
  
  # Make predictions on the training dataset
  train_preds <- predict(nn_model, train_nn)
  train_preds <- train_preds * (maxs[2] - mins[2]) + mins[2]
  train_err <- exp(train_preds) - exp(train$Price)
  train_abserr <- abs(train_err)
  train_percabserr <- train_abserr / exp(train$Price)
  train_mape <- mean(train_percabserr)
  #train_mape <- mean(abs(exp(train_nn$Price) - exp(train_preds)) / exp(train_nn$Price))
  
  # Make predictions on the testing dataset
  test_preds <- predict(nn_model, test_nn)
  test_preds <- test_preds * (maxs[2] - mins[2]) + mins[2]
  test_err <- exp(test_preds) - exp(test$Price)
  test_abserr <- abs(test_err)
  test_percabserr <- test_abserr / exp(test$Price)
  test_mape <- mean(test_percabserr)
  #test_mape <- mean(abs(exp(test_nn$Price) - exp(test_preds)) / exp(test_nn$Price))
  
  # Return the results
  c(num_neurons, train_mape, test_mape)
}

# Specify the number of neurons to test
num_neurons_list <- c(1, 5, 10, 20)

# Train and test the models with varying number of neurons
results <- t(sapply(num_neurons_list, train_and_test))

# Add column names to the results table
colnames(results) <- c("num_neurons", "train_mape", "test_mape")

# Print the results table
print(results)

#Now run the optimal neural network and print its summary
nn_Omodel <- nnet(Price ~ ., data = train_nn, size = 10, linout = F, decay=0.01, maxit=100)
summary(nn_Omodel)


#(2B)Below is the reason for what number of hidden neurons would be optimal.
'From the results above, number of hidden neurons when queals to 10 produces the output which has the low MAPE and does not require high computation like when hidden nuerons are 20'
#(2C) 
'I would use the optimal neural network from this model since it has lower MAPE when compared to the MAPE of the linear regression model in the part-1 above.'
#(2D)# Predict the prices of each diamond in the training dataset
train_nn$predicted_price <- predict(nn_Omodel, train_nn)

train_nn$predicted_price <- train_nn$predicted_price * (maxs[2] - mins[2]) + mins[2]
#(1I) # assuming the training dataset is called "diamonds_train"
recommended_diamond <- train_nn %>% mutate(predicted_price = (train_nn$predicted_price)) %>%
  filter(predicted_price <= log(12000)) %>%
  slice_max(Carat.Weight)
recommended_diamond$predicted_price <- exp(recommended_diamond$predicted_price)
# output the details of the recommended diamond
print(recommended_diamond)
'For Greg, I will suggest a diamond with 0.486 carat weight, good color, SI1 Clarity, ID symmetry and a GIA Report.'

#(2E)
'Building a neural network model using the nnet package in R typically involves the following steps:

Data preparation: This involves preparing the data that will be used to train and test the neural network. This may include tasks such as data cleaning, normalization or standardization, splitting the data into training and testing sets, and creating dummy variables for categorical predictors.

Model architecture: The next step is to specify the architecture of the neural network, including the number of hidden layers, the number of neurons in each layer, and the activation functions used in each layer. The nnet package provides a function called nnet() that allows you to specify these parameters.

Training the model: Once the architecture has been defined, the neural network can be trained using the training data. This involves updating the weights and biases of the neurons in the network through a process called backpropagation. The nnet() function in nnet package allows you to specify the number of iterations (epochs) to train the neural network.

Model evaluation: After the neural network has been trained, it is important to evaluate its performance on the test data. This involves using metrics such as accuracy, precision, recall, and F1 score to assess how well the model is able to predict outcomes on new, unseen data.

Model refinement: Depending on the performance of the neural network, it may be necessary to refine the model architecture or hyperparameters in order to improve its performance on the test data. This may involve experimenting with different activation functions, increasing the number of hidden layers or neurons, or changing the learning rate used in backpropagation.

Model deployment: Once the neural network has been trained and evaluated, it can be used to make predictions on new, unseen data. This may involve integrating the neural network into a larger software system or deploying it as a standalone application.

Overall, building a neural network model using the nnet package in R requires a solid understanding of neural network architecture, backpropagation, and model evaluation, as well as expertise in data preprocessing and analysis. It is a complex but powerful technique that can be used to solve a wide range of problems in fields such as finance, healthcare, and marketing'

