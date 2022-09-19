library(tidyverse)
library(mice)
library(VIM)
library(caret)
library(glmnet)
library(gbm)


# load data

train <- read_csv("train.csv")
test <- read_csv("test.csv")


# EDA 

aggr(train, labels = names(train), sortVars = TRUE)
matrixplot(train)
matrixplot(test)

# convert character variables to factor

train$location <- as.factor(train$location)
train$wind_gust_dir <- as.factor(train$wind_gust_dir)
train$wind_dir9am <- as.factor(train$wind_dir9am)
train$wind_dir3pm <- as.factor(train$wind_dir3pm)

test$location <- as.factor(test$location)
test$wind_gust_dir <- as.factor(test$wind_gust_dir)
test$wind_dir9am <- as.factor(test$wind_dir9am)
test$wind_dir3pm <- as.factor(test$wind_dir3pm)


# impute missing data with MICE

train <- train %>% 
  select(-evaporation, -sunshine)

test <- test %>% 
  select(-evaporation, -sunshine)


set.seed(100)
mids <- mice(train,
             m = 1,
             nnet.MaxNWts = 1800,
             seed = 1133)

comp <- complete(mids, 1)
write_csv(comp, "imptrain.csv")

aggr(comp, labels = names(train), sortVars = TRUE)

# imputation on test set

set.seed(1234)
midstest <- mice(test,
                 m = 1,
                 nnet.MaxNWts = 1800,
                 seed = 1122)

imptest <- complete(midstest, 1)


# partition data for cross validation

set.seed(222)
trainindex <- createDataPartition(comp$rain_tomorrow, p = 0.75, list = FALSE)
comptrain <- comp[trainindex,]
comptest <- comp[-trainindex,]

#comptrain$rain_tomorrow <- as.factor(comptrain$rain_tomorrow)

xtrain <- model.matrix(rain_tomorrow ~ ., data = comptrain)[,-1]
xtest <- model.matrix(rain_tomorrow ~ ., data = comptest)[,-1]
ytrain <- comptrain$rain_tomorrow
ytest <- comptest$rain_tomorrow


# lasso CV to find lambda

lam <- exp(seq(-15, 4, .05))

set.seed(123)
lassocv <- cv.glmnet(xtrain, ytrain, alpha = 1, lambda = lam)
plot(lassocv)

lassomod <- glmnet(xtrain, as.factor(ytrain), alpha = 1, 
                   lambda = lassocv$lambda.min,
                   family = "binomial")

lassopred <- predict(lassomod, s = lassocv$lambda.min, newx = xtest,
                     type = "response")

log_loss_calc <- function(preds, actual) {
  loss <- -((actual*log(preds)) + ((1-actual)*log(1-preds)))
  total <- sum(loss) / nrow(preds)
  return(total)
}

log_loss_calc(lassopred, ytest)
MLmetrics::LogLoss(lassopred, ytest)



# lasso model on test set

imptest$hat <- 0
testmat <- model.matrix(hat ~ ., data = imptest)[,-1]

yhattest <- predict(lassomod, s = lassocv$lambda.min, newx = testmat,
                    type = "response")

# create lasso csv

out <- data.frame(imptest$id, yhattest)
names(out)[1] <- "id"
names(out)[2] <- "rain_tomorrow"

write_csv(out, "rainsubmitlasso.csv")



# GBM model

set.seed(8888)
fit_gbm <- gbm(rain_tomorrow ~ ., data = comptrain,
               distribution = "bernoulli",
               shrinkage = .01,
               n.trees = 4000,
               verbose = FALSE)

gbmpred <- predict(fit_gbm, newdata = comptest, type = "response")

log_loss_calc(gbmpred, ytest)
MLmetrics::LogLoss(gbmpred, ytest)


# gbm with 12 fold CV and 10 learning rates


comptrain$date <- as.numeric(comptrain$date)
comptest$date <- as.numeric(comptest$date)
comp$date <- as.numeric(comp$date)

s <- seq(.001, .2, length = 10)
perform <- c()
for (j in 1:length(s)) {
  set.seed(12)
  fold <- sample(rep(1:12, each = 2137), replace = FALSE)
  yhat <- rep(NA, nrow(comptrain))
  for (i in 1:12) {
    fit_gbm <- gbm(rain_tomorrow ~ ., data = comptrain[fold != i,],
                   distribution = "bernoulli",
                   shrinkage = s[j],
                   n.trees = 3000,
                   verbose = FALSE)
    yhat[fold == i] <- predict(fit_gbm, comptrain[fold == i,],
                               type = "response")
  }
  perform[j] <- MLmetrics::LogLoss(yhat, ytrain)

}

chart <- data.frame(s, perform)

ggplot(chart, aes(s, perform)) +
  geom_point() +
  geom_line()

# best learning rate

which.min(perform)
shrink <- s[5]

# optimal number of trees

set.seed(77777)
gbm1 <- gbm(rain_tomorrow ~ ., data = comptrain,
            distribution = "bernoulli",
            shrinkage = shrink,
            n.trees = 3000,
            verbose = FALSE,
            cv.folds = 12)

trees <- gbm.perf(gbm1, method = "cv")
trees


# final model on whole training set

set.seed(55555)
finalgbm <- gbm(rain_tomorrow ~ ., data = comp,
                distribution = "bernoulli",
                shrinkage = shrink,
                n.trees = 2709,
                verbose = FALSE)

summary(finalgbm)


gbmhat <- predict(finalgbm, newdata = imptest, type = "response")


# create gbm CSV

gbm_out <- data.frame(imptest$id, gbmhat)
names(gbm_out)[1] <- "id"
names(gbm_out)[2] <- "rain_tomorrow"

write_csv(gbm_out, "rainsubmitgbm.csv")
