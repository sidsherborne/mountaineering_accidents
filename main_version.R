library(readxl)
library(magrittr)
library(dplyr)
library(keras)



# Downloading Data From GitHub
data <- rio::import("https://github.com/ecaroom/climbing-accidents/raw/master/_github-AAC_accidents_tagged_data.xlsx")

# Replacing NA values with 0.
data <- data %>% mutate_all(function(x) ifelse(is.na(x), 0, x))

# Dropping some columns.
data <- data %>% select(-c(`Search Column\r\n\r\n`, ID, `Accident Title`, Text, `Tags Applied`, `COUNT OF TAGS`, Serious, Minor, `Head / Brain Injury`))

# Removing one missing value.
data <- data %>% mutate(`Publication Year` = ifelse(0, NA, `Publication Year`))

# Binding with Train Data
train_complete <- data 
train_complete <- train_complete %>% mutate_all(as.numeric) 

# Splitting into data and labels.
train_data <- train_complete %>% select(-"Deadly")
train_labels <- train_complete %>% use_series("Deadly") 

# Data transformation some all values between 0 and 1.  ----------------------------------------------------
train_data <- train_data %>% mutate_all(function(x) x/max(x, na.rm = TRUE))

# Train data to data matrix.
x_train <- train_data %>% data.matrix() 
y_train <- train_labels

# Model -------------------------------------------------------------------
network <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(86),  kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
  layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid",  kernel_initializer = "uniform")

# Why not using Stochastic Gradient descent and defining a learning rate?
network %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


# Run 
network %>% fit(x_train, y_train, epochs = 10, batch_size = 1, validation_split = 0.3)

# Setup lime::model_type() function for keras
model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  "classification"
}

# Setup lime::predict_model() function for keras
predict_model.keras.engine.sequential.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
}

# Lots of gc()s for my dodgey old laptop.
gc()

# Setting up lime explainer
explainer <- lime::lime(
  x              = train_data, 
  model          = network, 
  bin_continuous = FALSE
)

# Explaining the classification of the first 10 cases.
gc()
explanation <- lime::explain (
  train_data[1:100, ], # Just to show first 10 cases
  explainer    = explainer, 
  n_labels     = 1, # explaining a `single class`(Polarity)
  n_features   = 20, # returns top four features critical to each case
  kernel_width = 0.5)
gc()

# Plotting the most important features for the first 10 cases.
output <- lime::plot_features(explanation, ncol = 2)
ggplot2::ggsave("output.pdf", plot = output, paper = "a4", device = "pdf", width = 210, height = 297, units = "mm")

# Plotting the frequency of important features of the first 10 cases.
output_all <- lime::plot_explanations (explanation)
ggplot2::ggsave("output_all.pdf", plot = output_all, paper = "a4", device = "pdf", width = 210, height = 297, units = "mm")
