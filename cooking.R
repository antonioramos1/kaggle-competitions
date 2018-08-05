library(jsonlite)
library(magrittr)
library(dplyr)
library(tm)
library(CatEncoders)
library(randomForest)
library(caret)
library(readr)

setwd("C:/Users/heret/Downloads/whats-cooking/")
train <- fromJSON("train.json")
test <- fromJSON("test.json")

#preprocessing and structuring data
ingredients_corpus <- c(VCorpus(VectorSource(train$ingredients)), VCorpus(VectorSource(test$ingredients)))
ingredients_corpus <- tm_map(ingredients_corpus, tolower)
ingredients_corpus <- tm_map(ingredients_corpus, removePunctuation)
ingredients_corpus <- tm_map(ingredients_corpus, removeWords, stopwords("english"))
ingredients_corpus <- tm_map(ingredients_corpus, removeNumbers)
ingredients_corpus <- tm_map(ingredients_corpus, stripWhitespace)
ingredients_corpus <- tm_map(ingredients_corpus, stemDocument)
ingredients_corpus <- tm_map(ingredients_corpus, PlainTextDocument)

ingredients_dtm_train <- DocumentTermMatrix(ingredients_corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
ingredients_sparse <- removeSparseTerms(ingredients_dtm_train, 0.999) #1-5/nrow(ingredients_dtm_train)) #removes words with freq < 5

ingredients_df <- as.data.frame(as.matrix(ingredients_sparse))
ingredients_df$total_ingredients  <- rowSums(ingredients_df) #counting the ingredients per recipy
ingredients_df$cuisine <- as.factor(c(train$cuisine, rep("italian", nrow(test))))
rm(ingredients_corpus, ingredients_sparse, ingredients_dtm_train)

train_df  <- ingredients_df[1:nrow(train), ]
test_df <- ingredients_df[-(1:nrow(train)), ]

#train test split to evaluate model
# index <- createDataPartition(y = train_df$cuisine, p = 0.8, list = FALSE)
# train_X <- train_df[index,]
# test_X <- train_df[-index,]
# 
# set.seed(2018)
# start_time <- Sys.time()
# rforest_eval <- randomForest(cuisine ~ ., data=train_X, importance=TRUE, ntree=150)
# end_time <- Sys.time()
# print(end_time - start_time)
# 
# predictions <- predict(rforest_eval, newdata = test_X, type = "class")
# conf_matrix <- confusionMatrix(predictions, test_X$cuisine)
# print(conf_matrix) #0.76 accuracy

#model on train data
set.seed(2018)
start_time <- Sys.time()
rforest <- randomForest(cuisine ~ ., data=train_df, importance=TRUE, ntree=150)
end_time <- Sys.time()
print(end_time - start_time)

predictions <- predict(rforest, newdata = test_df, type = "class")
my_submission <- tibble(as.vector(test$id), predictions)
names(my_submission) <- c("id", "cuisine")
write_csv(my_submission, path="cooking.csv")  #0.75 accuracy LB