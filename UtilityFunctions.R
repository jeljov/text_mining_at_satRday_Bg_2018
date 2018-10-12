## This script is created by Jelena Jovanovic (http://jelenajovanovic.net/), 
## for the Text Mining workshop held at SatRday Belgrade 2018 event 
## (https://belgrade2018.satrdays.org/#workshops)


## Function for creating a feature data frame out of
## - a DTM, represented in the form of quanteda's dfm, and 
## - a vector of class labels
create_feature_df <- function(train_dfm, class_labels) {
  
  train_df <- convert(train_dfm, "data.frame")
  # The 'convert' f. from quanteda adds 'document' as the 1st feature (column)
  # in the resulting data frame. It needs to be removed before the data frame 
  # is used for training.
  if ((names(train_df)[1] == 'document') & (class(train_df[,1])=='character'))
    train_df <- train_df[, -1]
  
  # Check if there are documents that have 'lost' all their words, that is,
  # if there are rows with all zeros
  doc_word_cnt <- rowSums(train_df)
  zero_word_docs <- which(doc_word_cnt == 0)
  # If there are zero-word rows, remove them
  if (length(zero_word_docs) > 0) {
    print(paste("Number of documents to remove due to sparsity:", length(zero_word_docs)))
    train_df <- train_df[-zero_word_docs,]
    class_labels <- class_labels[-zero_word_docs]
  }
  
  # Assure that column names are regular R names
  require(janitor)
  train_df <- clean_names(train_df)
  
  # Combine class labels and the features 
  cbind(Label = class_labels, train_df)
  
}

## Function for plotting the distribution of word weights
plot_word_weight_distr <- function(wweights, lbl, bin_width = 0.1) {
  require(ggplot2)
  ggplot(data = data.frame(weights = wweights), mapping = aes(x = weights)) + 
    geom_histogram(aes(y=..density..),  # Histogram with density instead of count on y-axis
                   binwidth=bin_width,
                   colour="black", fill="white") +    
    geom_density(alpha=.2) +
    xlab(lbl) +
    theme_bw() 
}

## Function for performing 5-fold cross validation on the given training data set
## (train_data) using the specified ML algorithm (ml_method). 
## Cross-validation is done in parallel on the specified number (nclust) of logical cores.
## The grid_spec serves for passing the grid of values to be used in tuning one or more 
## parameter(s) of the ML method.
## The ntree parameter can be used to set the number of trees when Random Forest is used.
cross_validate_classifier <- function(seed,
                                      nclust, 
                                      train_data, 
                                      ml_method,
                                      grid_spec,
                                      ntree = 1000) { 
  require(caret)
  require(doSNOW)
  
  # Setup the CV parameters
  cv_cntrl <- trainControl(method = "cv", 
                           number = 5, 
                           search = "grid",
                           summaryFunction = twoClassSummary, # computes sensitivity, specificity, AUC
                           classProbs = TRUE, # required for the twoClassSummary f.
                           allowParallel = TRUE) # default value, but to emphasize the use of parallelization
  
  # Create a cluster to work on nclust logical cores;
  # what it means (simplified): create nclust instances of RStudio and 
  # let caret use them for the processing 
  cl <- makeCluster(nclust, 
                    type = "SOCK") # SOCK stands for socket cluster
  registerDoSNOW(cl)
  
  # Track the time of the code execution
  start_time <- Sys.time()
  
  set.seed(seed)
  if (ml_method=="rpart")
    model_cv <- train(x = train_data[,names(train_data) != 'Label'],
                      y = train_data$Label,
                      method = 'rpart', 
                      trControl = cv_cntrl, 
                      tuneGrid = grid_spec, 
                      metric = 'ROC')
  if (ml_method=="ranger") {
    require(ranger)
    model_cv <- train(x = train_data[,names(train_data) != 'Label'],
                      y = train_data$Label,
                      method = 'ranger', 
                      trControl = cv_cntrl, 
                      tuneGrid = grid_spec, 
                      metric = 'ROC',
                      num.trees = ntree,
                      importance = 'impurity',
                      verbose = TRUE)
  }
  
  # Processing is done, stop the cluster
  stopCluster(cl)
  
  # Compute and print the total time of execution
  total_time <- Sys.time() - start_time
  print(paste("Total processing time:", total_time))
  
  # Return the built model
  model_cv
  
}

## Function for calculating relative (normalized) term frequency (TF)
relative_term_frequency <- function(row) { # in DTM, each row corresponds to one document 
  row / sum(row)
}

## Function for calculating inverse document frequency (IDF)
## Formula: log(corpus.size/doc.with.term.count)
inverse_doc_freq <- function(col) { # in DTM, each column corresponds to one term (feature) 
  corpus.size <- length(col) # the length of a column is in fact the number of rows (documents) in DTM
  doc.count <- length(which(col > 0)) # number of documents that contain the term
  log10(corpus.size / doc.count)
}

## Function for calculating TF-IDF
tf_idf <- function(x, idf) {
  x * idf
}


## The function extracts some basic evaluation metrics from the model evaluation object
## produced by the confusionMatrix() f. of the caret package
get_eval_measures <- function(pred_model, test_df, metrics) {
  eval_measures <- list()
  
  model_eval <- confusionMatrix(data = predict(pred_model, test_df), 
                                reference = test_df$Label)
  by_class_measures <- names(model_eval$byClass)
  overall_measures <- names(model_eval$overall)
  for(m in metrics) {
    if(m %in% by_class_measures)
      eval_measures[[m]] <- as.numeric(model_eval$byClass[m])
    else if(m %in% overall_measures)
      eval_measures[[m]] <- as.numeric(model_eval$overall[m])
  }
  if(('AUC' %in% metrics) | ('ROC' %in% metrics))
    eval_measures['AUC'] <- compute_auc(pred_model, test_df)
    
  unlist(eval_measures)
}


compute_auc <- function(pred_model, test_df) {
  preds_prob <- predict(pred_model, test_df, type = 'prob')
  require(pROC)
  # the 1st argument of the roc f. is the probablity of the 
  # positive class, which is, by default, the class that 
  # corresponds to the level 1 of the class (factor) variable
  # (N.B. roc f. uses terms 'cases' and 'controls' to refer to
  # the positive and negative classes, respectively)
  roc <- roc(predictor=preds_prob[,2], response=test_df$Label)
  plot.roc(roc, print.auc = TRUE)
  roc$auc
}
