# Using this example
# https://liuyanguu.github.io/post/2019/07/18/visualization-of-shap-for-xgboost/

# Similar Packages:
# iml: https://cran.r-project.org/web/packages/iml/vignettes/intro.html



library(SHAPforxgboost)
library(ggplot2)
library(xgboost)

library(readxl)
library(magrittr)
library(dplyr)

# Downloading Data From GitHub
data <- rio::import("https://github.com/ecaroom/climbing-accidents/raw/master/_github-AAC_accidents_tagged_data.xlsx")

# Replacing NA values with 0.
data <- data %>% mutate_all(function(x) ifelse(is.na(x), 0, x))

# Keeping Deadly or Serious Accidents
data <- data %>% filter(Deadly==1 | Serious == 1)

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

# Data transformation some all values between 0 and 1.
#train_data <- train_data %>% mutate_all(function(x) x/max(x, na.rm = TRUE))

# Train data to data matrix.
x_train <- train_data %>% data.matrix() 
y_train <- train_labels %>% data.matrix()


# hyperparameter tuning results
param_dart <- list(objective = "binary:logistic",  # For regression
                   # booster = "dart",
                   nrounds = 10,
                   eta = 0.018,
                   max_depth = 10,
                   gamma = 0.009,
                   subsample = 0.98,
                   colsample_bytree = 0.86
)

mod <- xgboost::xgboost(data = x_train, 
                        label = y_train, 
                        xgb_param = param_dart, nrounds = param_dart$nrounds,
                        verbose = FALSE, nthread = parallel::detectCores() - 2,
                        early_stopping_rounds = 8)

# To return the SHAP values and ranked features by mean|SHAP|
shap_values <- shap.values(xgb_model = mod, X_train = x_train)

# The ranked features by mean |SHAP|
shap_values$mean_shap_score


# To prepare the long-format data:
shap_long <- shap.prep(xgb_model = mod, X_train = x_train)
# is the same as: using given shap_contrib
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = x_train)

# **SHAP summary plot**
shap_final <- shap.plot.summary(shap_long)

ggplot2::ggsave("output_shap_binary.png", plot = shap_final, device = "png", width = 210, height = 297, units = "mm")
