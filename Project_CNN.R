install.packages("plyr")
install.packages("mxnet")
library(caret)
library(plyr)
require(mxnet)
setwd("E:/MSIS/Sem 3/ADS/Project")

#import the semicolon dataset
semeion_data <- read.table("semeion.txt")
write.csv(semeion_data, file="Handwritten.csv")
str(semeion_data)
semeion_data
class(semeion_data)

semeion_data$digit <- NA
str(semeion_data)
semeion_data$digit
semeion_data$digit[which(semeion_data$V257 == 1)] <- 0
semeion_data$digit[which(semeion_data$V258 == 1)] <- 1
semeion_data$digit[which(semeion_data$V259 == 1)] <- 2
semeion_data$digit[which(semeion_data$V260 == 1)] <- 3
semeion_data$digit[which(semeion_data$V261 == 1)] <- 4
semeion_data$digit[which(semeion_data$V262 == 1)] <- 5
semeion_data$digit[which(semeion_data$V263 == 1)] <- 6
semeion_data$digit[which(semeion_data$V264 == 1)] <- 7
semeion_data$digit[which(semeion_data$V265 == 1)] <- 8
semeion_data$digit[which(semeion_data$V266 == 1)] <- 9


semeion_data$digit
head(semeion_data)
tail(semeion_data)
dim(semeion_data)

semeion_data <- semeion_data[,-c(257,258,259,260,261,262,263,264,265,266)]

dim(semeion_data)
str(semeion_data)

#semeion_data$digit <- factor(semeion_data$digit)
str(semeion_data)
?
ddply(semeion_data,~digit,summarise, freq=length(digit))
dim(semeion_data)

shuffled <- semeion_data[sample(1:1593),]

# Train-test split
train<- shuffled[1:1280, ]
ncol(train)
test <- shuffled[1281:1593, ]

ddply(train,~digit,summarise, freq=length(digit))
ddply(test,~digit,summarise, freq=length(digit))

#splitting data
index <- sample(1:nrow(shuffled),round(0.8*nrow(shuffled)))
train_data <- shuffled[index,]
test_data <- shuffled[-index,]
shuffled$digit
train$digit

ncol(test_data)

ddply(train_data,~digit,summarise, freq=length(digit))
ddply(test_data,~digit,summarise, freq=length(digit))

#setting up train and test data
#train_data<- data.matrix(train_data)
ncol(train_data)
train_input <- t(train_data[,-257])
train_output <- train_data[,257]
dim(train_input) <- c(16,16,1,ncol(train_input))

#test_data<- data.matrix(test_data)
test_output <- test_data[,257]
test_input <- t(test_data[,-257])
test_output <- test_data[,257]
dim(test_input) <- c(16,16,1,ncol(test_input))

#setting up symbolic model
# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(3,3), num_filter=10)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(3,3), num_filter=80)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# third conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(3,3), num_filter=80)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))



# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)



# Set seed for reproducibility
#mx.set.seed(100)

# Device used. CPU in my case.
devices <- mx.cpu()




mx.set.seed(1)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet,  X = train_input,
                                     y = train_output,
                                     ctx=devices, num.round=200, array.batch.size=100,
                                     learning.rate=0.05 , momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))



# Predict labels
predicted <- predict(model, test_input)
# Assign labels
predicted_labels <- max.col(t(predicted)) - 1
# Get accuracy
table(test_output, predicted_labels)
sum(diag(table(test_output, predicted_labels)))/319


confusionMatrix(as.numeric(test_output), as.numeric(predicted_labels))
