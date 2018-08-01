# Credit to:
# https://www.kaggle.com/startupsci/titanic-data-science-solutions   Basic feature engineering basic modeling  
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy  A look into hyperparameterization with sklearn  
# https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever/notebook  Advanced feature engineering covering group/family survival assumption feature

# Other references:
# http://www.endmemo.com/program/R/grep.php  String treatment in R 
# https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/ Support Vector Machines  
# https://medium.com/deep-math-machine-learning-ai/chapter-3-support-vector-machine-with-math-47d6193c82be Support Vector Machines with math

# "Translation" to R from the python notebook. LB score = 0.7894
# Family and group survival feature has not been included

library(dplyr)
library(magrittr)
library(readr)
library(e1071)

train = read_csv("C:/Users/heret/Downloads/titanic/train.csv")
test = read_csv("C:/Users/heret/Downloads/titanic/test.csv")

all_data = bind_rows(train, test)
ntrain = nrow(train)
ntest = nrow(test)

# Discretising Sex variable
all_data %<>%
  mutate("Sex" = ifelse(Sex == "female", 1, 0))

# Replace Age NaNs by imputing them based on Sex and Pclass median
guess_ages = matrix(0, 2, 3)

for (i in seq(0,1)) {
  for (j in seq(3)) {
    guess_tbl <- all_data %>% 
      filter(Sex == i & Pclass == j & !is.na(Age)) %>% 
      select(Age)
    age_guess <-  sapply(guess_tbl, median)
    
    guess_ages[i+1,j] = round((age_guess/0.5 +0.5) * 0.5)
  }
}

for (i in seq(0,1)) {
  for (j in seq(3)) {
    all_data[(is.na(all_data$Age)) & (all_data$Sex == i) & all_data$Pclass == j, "Age"] <- guess_ages[i+1,j]
  
  }
}
all_data$Age <- round(all_data$Age)

# Replace Embarked NaNs with the mode
most_embarked = tail(names(sort(table(all_data["Embarked"]))), 1)
all_data$Embarked[is.na(all_data$Embarked)] <- most_embarked

# Encoding variable
all_data$Embarked <- as.numeric(as.factor(all_data$Embarked))

# Replacing Fare NaN with median
median_fare <- median(all_data$Fare, na.rm = TRUE)
all_data$Fare[is.na(all_data$Fare)] <- median_fare

# Creating a new IsAlone varible
all_data$FamilySize <- all_data$SibSp + all_data$Parch + 1
all_data$IsAlone <- 0
all_data[all_data$FamilySize == 1, "IsAlone"] <- 1

# Creating a new variable "LastName" so it can help us identify families
all_data$LastName <- sapply(strsplit(all_data$Name, split=","), head, 1)


# Creating a new Title variable
all_data$Title <- sapply(all_data$Name,function(x) gsub('(.*, )|(\\..*)', '', x)) #regex from https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

all_data$Title <- sapply(all_data$Title, function(x) gsub("(the Countess|Lady|Capt|Col|Don|Dr|Major|Rev|Sir|Jonkheer|Dona)", "Rare", x))
all_data$Title <- sapply(all_data$Title, function(x) gsub("(Ms|Mlle)", "Miss", x))
all_data$Title <- sapply(all_data$Title, function(x) gsub("(Mme)", "Mrs", x))
# table(all_data$Title) Discrepancies against Python result (?)
all_data$Title <- as.numeric(factor(all_data$Title, levels = c("Mr","Miss","Mrs","Master","Rare"), labels=c(1,2,3,4,5)))

# Binning Fare variable
all_data$FareBinned <- cut(all_data$Fare, breaks=5)
all_data$FareBinned<- as.numeric(as.factor(all_data$FareBinned))

# Binning Age variable
all_data$AgeBinned <- cut(all_data$Age, breaks=5)
all_data$AgeBinned <- as.numeric(as.factor(all_data$AgeBinned))

# Removing unnecessary columns
all_data <- select(all_data, -Name, -PassengerId, -Ticket, -Cabin, -Fare, -Age, -SibSp, -Parch, -LastName, -FamilySize)

train_X <- select(all_data, -Survived)[1:ntrain,]
train_y <- train$Survived
test_X <- select(all_data, -Survived)[(ntrain+1):nrow(all_data),]
#print(paste("train_X: ", dim(train_X), "train_y: ", dim(train_y), "test_X: ", dim(test_X), sep=" "))

# SVM
model <- svm(x=train_X, y=train_y, type ="C-classification")
predictions <- predict(model, test_X)
my_submission <- tibble(as.vector(test$PassengerId), predictions)
names(my_submission) <- c("PassengerId", "Survived")
write_csv(my_submission, path="titanic.csv")
