#EDA Lab Assignment 4
#21BDS0153
#VARSHITH N

library(tidyverse)
library(caret)
library(ggplot2)
library(corrplot)
library(FactoMineR)
library(missMDA)
library(randomForest)
library(GGally)


file_path <- "dataset.csv"  # Adjust if needed
df <- read.csv(file_path, stringsAsFactors = FALSE, na.strings = c("", "NA"))

unnecessary_cols <- c("rownames", "state")
df <- df %>% select(-any_of(unnecessary_cols))

binary_cols <- c("speed65", "speed70", "drinkage", "alcohol", "enforce")
for (col in binary_cols) {
  if (col %in% colnames(df)) {
    df[[col]] <- ifelse(df[[col]] == "yes", 1,
                        ifelse(df[[col]] == "no", 0, NA))
  }
}



df <- df %>% mutate(across(where(is.character), as.numeric))

df <- df[, sapply(df, function(col) length(unique(col)) > 1)]

#missing values
for (col in colnames(df)) {
  if (is.numeric(df[[col]])) {
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
  }
}


df <- na.omit(df)

#correlation matrix
numeric_cols <- df %>% select(where(is.numeric))
numeric_cols <- numeric_cols[, sapply(numeric_cols, function(col) sd(col, na.rm = TRUE) > 0)]
if (ncol(numeric_cols) > 1) {
  cor_matrix <- cor(numeric_cols, use = "complete.obs")
  corrplot(cor_matrix, method = "color", tl.col = "black", tl.cex = 0.8)
}

#histogram
if ("fatalities" %in% colnames(df)) {
  ggplot(df, aes(x = fatalities)) +
    geom_histogram(bins = 30, fill = "red", color = "black", alpha = 0.7) +
    ggtitle("Distribution of Fatalities") +
    xlab("Fatalities") + ylab("Frequency")
}

#1D/2D/ND visualization: 
#Pairplot
num_cols <- c("miles", "fatalities", "seatbelt", "income", "age")
num_cols <- intersect(num_cols, colnames(df))
if (length(num_cols) > 1) {
  ggpairs(df[num_cols])
}

#time series analysis
if ("year" %in% colnames(df)) {
  df %>%
    group_by(year) %>%
    summarise(avg_fatalities = mean(fatalities, na.rm = TRUE)) %>%
    ggplot(aes(x = year, y = avg_fatalities)) +
    geom_line(color = "blue") + geom_point() +
    ggtitle("Fatalities Over Time") +
    xlab("Year") + ylab("Average Fatalities")
}

#data modeling
df_model <- df %>% select(-fatalities)
target <- df$fatalities

df_model <- df_model[, sapply(df_model, function(x) sd(x) > 0)]

#data
preProcValues <- preProcess(df_model, method = c("center", "scale"))
df_scaled <- predict(preProcValues, df_model)

#missing values
if (any(is.na(df_scaled))) {
  df_scaled <- imputePCA(df_scaled, ncp = 5)$completeObs
}

#PCA
pca_result <- PCA(df_scaled, graph = FALSE)
pca_data <- as.data.frame(pca_result$ind$coord[, 1:5])

#train test split
set.seed(42)
train_index <- createDataPartition(target, p = 0.8, list = FALSE)
train_data <- pca_data[train_index, ]
test_data <- pca_data[-train_index, ]
train_target <- target[train_index]
test_target <- target[-train_index]

#random forest model
set.seed(42)
rf_model <- randomForest(x = train_data, y = train_target, ntree = 100)

#evaluate the model
predictions <- predict(rf_model, test_data)
mae <- mean(abs(predictions - test_target))
mse <- mean((predictions - test_target)^2)
r2 <- cor(predictions, test_target)^2

#print results
cat("Model Evaluation:\n")
cat("MAE:", round(mae, 4), "\n")
cat("MSE:", round(mse, 4), "\n")
cat("R² Score:", round(r2, 4), "\n")

