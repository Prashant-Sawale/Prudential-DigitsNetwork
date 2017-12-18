library(Metrics)
library(nnet)
library(NeuralNetTools)
library(readr)
library(stringr)
library(caret)
library(car)
library(neuralnet)
library(forecast)

dfq <- read.csv(file = "Filtered_data.csv")
#set.seed(101) # Set Seed so that same sample can be reproduced in future also
# Now Selecting 75% of data as sample from total 'n' rows of the data  
#sample <- sample.int(n = nrow(dfq), size = floor(.5*nrow(dfq)), replace = F)
#df <- dfq[sample, ]
#df_not  <- data[-sample, ]
df <- dfq[,c("Medical_History_41.1","Product_Info_3.6", "Insurance_History_1.1",
             "Employment_Info_3.1", "Medical_History_3.2", "InsuredInfo_6.1",
             "Medical_History_13.1","Medical_History_23.1", "Medical_History_39.1",
             "Medical_History_4.1", "Medical_Keyword_25", "Medical_Keyword_37",
             "Medical_Keyword_48", "Medical_History_1","Family_Hist_4",
             "Medical_Keyword_23", "Product_Info_4" , "BMI" , "Ins_Age" , "Response")]


str(df)


str(df)

Target<-(df$Response/8)
tail(df$Response)
head(df$Response)
tail(Target)
head(Target)
df_s <- subset(df, select = -c(Response) )




#require(caret)
#trans = preProcess(df_s, 
#                 method=c("BoxCox", "center", 
#                        "scale", "pca"))
#PC = predict(trans, df_s)
#str(PC)
#class(PC)
maxs <- apply(df_s, 2, max) 
mins <- apply(df_s, 2, min)

scaled <- as.data.frame(scale(df_s, center = mins, scale = maxs - mins))
str(scaled)
class(scaled)



dat<-cbind(scaled,Target)
str(dat)



# Train-test split
set.seed(301)
df1 <- sample(1:nrow(dat),round(0.8*nrow(dat)))
train <- dat[df1,]
test <- dat[-df1,]
test_df<-subset(test, select = -c(Target) )
#custom <- function(x) {log(1+exp(x))}

# NN training
n <- names(train)
f <- as.formula(paste("Target ", paste(n[!n %in% "Target"], collapse = " + "), sep=" ~ "))


trc <- trainControl (method="repeatedcv", number=2, repeats=5)
model.mlp <- train(f,data=train, method='nnet', MaxNWts=15000, maxit=50, 
                   trace = FALSE,tuneGrid = expand.grid(.size=seq(1,20,by=1),.decay=0), trControl=trc)

model.mlp$results
model.mlp$bestTune 
plot(model.mlp)


# Neural net with hidden layer = 9, act.fct = "logistic" and threshold = 0.25
nn <- neuralnet(f,data=train,hidden= c(9,4),threshold = 0.2, linear.output=F,lifesign = "full")

plot(nn)

#predicting output
pr.nn <- compute(nn,test_df)
pr.nn_ <- pr.nn$net.result*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
test.r <- (test$Target)*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
Result<- rmse(pr.nn_,test.r)
Result




#Neural net with custom ReLU(Softplus) activation function

custom <- function(x) {log(1+exp(x))}
nn_relu <- neuralnet(f,data=train,hidden= c(9,4),threshold = 0.7, linear.output=T,
                     lifesign = "full", act.fct = custom )

plot(nn_relu)

#predicting output
pr.nn_relu <- compute(nn_relu,test_df)
pr.nn_R <- pr.nn_relu$net.result*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
test.r <- (test$Target)*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
Result_relu<- rmse(pr.nn_R,test.r)
Result_relu









#With tanH 
nn_tanh <- neuralnet(f,data=train,hidden= c(9,4),threshold = 0.3, linear.output=T,
                     lifesign = "full", act.fct = "tanh" )

plot(nn_tanh)

#predicting output
pr.nn_tanh <- compute(nn_tanh,test_df)
pr.nn_T <- pr.nn_tanh$net.result*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
test.t <- (test$Target)*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
Result_tanh<- rmse(pr.nn_T,test.r)
Result_tanh


#With tanH and rprop-
nn_prop <- neuralnet(f,data=train,hidden= c(9,4),threshold = 0.3, linear.output=T,
                     lifesign = "full", act.fct = "tanh", algorithm = "rprop-" )

plot(nn_prop)

#predicting output
pr.nn_prop <- compute(nn_prop,test_df)
pr.nn_P <- pr.nn_prop$net.result*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
test.t <- (test$Target)*(max(df["Response"])-min(df["Response"]))+min(df["Response"])
Result_rprop<- rmse(pr.nn_P,test.r)
Result_rprop






