setwd('E:\\TCGA\\Final_Project\\omic')
rm(list=ls())
meta=read.csv("meta.csv",header = T,stringsAsFactors = F)
exprSet<-read.csv("exprSet.csv",header = T,check.names = F) 
row.names(exprSet)<-make.names(exprSet[,1],TRUE)
exprSet<-exprSet[,-1]
##转录组
#LASSO
x=t(exprSet)
y=meta$event
library(glmnet)
library(Matrix)
cv_fit<-cv.glmnet(x=x,y=y)
plot(cv_fit)
fit<-glmnet(x=x,y=y)
plot(fit,xvar="lambda")
model_lasso_min <- glmnet(x=x, y=y,lambda=cv_fit$lambda.min)
choose_gene_min=rownames(model_lasso_min$beta)[as.numeric(model_lasso_min$beta)!=0]
length(choose_gene_min)
lasso.prob <- predict(cv_fit, newx=x, s=c(cv_fit$lambda.min,cv_fit$lambda.1se) )
re=cbind(y ,lasso.prob)
re=as.data.frame(re)
colnames(re)=c('event','prob_min','prob_1se')
re$event=as.factor(re$event)
tpr_min = performance(pred_min,"tpr")@y.values[[1]]
dat = data.frame(tpr_min = perf_min@y.values[[1]],
                 fpr_min = perf_min@x.values[[1]])
metaTrain=read.csv("meta_train.csv",header = T,stringsAsFactors = F)
exprSetTrain<-read.csv("exprSet_train.csv",header = T,check.names = F) 
row.names(exprSetTrain)<-make.names(exprSetTrain[,1],TRUE)
exprSetTrain<-exprSetTrain[,-1]
metaTest=read.csv("meta_test.csv",header = T,stringsAsFactors = F)
exprSetTest<-read.csv("exprSet_test.csv",header = T,check.names = F) 
row.names(exprSetTest)<-make.names(exprSetTest[,1],TRUE)
exprSetTest<-exprSetTest[,-1]
train <- exprSetTrain
test <- exprSeTset
train_meta <- metaTrain
test_meta <- metaTest
x = t(train)
y = train_meta$event
cv_fit <- cv.glmnet(x=x, y=y)
plot(cv_fit)
model_lasso_min <- glmnet(x=x, y=y,lambda=cv_fit$lambda.min)
choose_gene_min=rownames(model_lasso_min$beta)[as.numeric(model_lasso_min$beta)!=0]
lasso.prob <- predict(cv_fit, newx=t(test), s=c(cv_fit$lambda.min,cv_fit$lambda.1se) )
re=cbind(event = test_meta$event ,lasso.prob)
re=as.data.frame(re)
colnames(re)=c('event','prob_min','prob_1se')
re$event=as.factor(re$event)
head(re)
library(ROCR)
pred_min <- prediction(re[,2], re[,1])
auc_min = performance(pred_min,"auc")@y.values[[1]]
perf_min <- performance(pred_min,"tpr","fpr")
ggplot() + 
  geom_line(data = dat,aes(x = fpr_min, y = tpr_min),color = "blue") + 
  geom_line(aes(x=c(0,1),y=c(0,1)),color = "grey")+
  theme_bw()+
  annotate("text",x = .75, y = .25,
           label = paste("AUC of min = ",round(auc_min,2)),color = "blue")+
  scale_x_continuous(name  = "fpr")+
  scale_y_continuous(name = "tpr")
#随机森林
library(randomForest)
library(ROCR)
library(Hmisc)
train <- exprSetTrain
test <- exprSetTest
train_meta <- metaTrain
test_meta <- metaTest
x = t(train)
y = train_meta$event
rf_output=randomForest(x=x, y=y,importance = TRUE, ntree = 10001, proximity=TRUE )
varImpPlot(rf_output,type=2,n.var=30,scale=FALSE,main="Variable Importance for Top 30 Predictors",cex=.7)
choose_gene=rownames(tail(rf_importances[order(rf_importances[,2]),],30))
x_1=t(test)
y_1=test$meta$event
rftest.prob <- predict(rf_output, x_1)
retest=cbind(y_1 ,rftest.prob)
pred <- prediction(re[,2], re[,1])
auc = performance(pred,"auc")@y_1.values[[1]]
##临床
library(survival)
library(survminer)
library(My.stepwise)
vl<-colnames(meta)
My.stepwise.coxph(Time ="time",
                  Status ="event",
                  variable.list = vl,
                  data=dat)
model=coxph(formula = Surv(time, event)~time+stage+fvc,data=meta)
ggforest(model,data = meta)
##SNP
meta=read.csv("meta.csv",header = T,stringsAsFactors = F)
SNPSet<-read.csv("SNP.csv",header = T,check.names = F) 
row.names(SNPSet)<-make.names(SNPSet[,1],TRUE)
SNPSet<-SNPSet[,-1]
library(e1071)
train <- SNPSetTRAIN
test <- SNPSetTEST
train_meta <- metaTRAIN
test_meta <- metaTEST
x=t(train)
X_TEST=t(test)
y=train_meta$event
Y_TEST=test_meta$event
rf_output=randomForest(x=x, y=y,importance = TRUE, ntree = 10001, proximity=TRUE )
varImpPlot(rf_output,type=2,n.var=30,scale=FALSE,main="Variable Importance for Top 30 Predictors",cex=.7)
rf.prob <- predict(rf_output, X_TEST)
re=cbind(Y_TEST ,rf.prob)
auc = performance(pred,"auc")@Y_TEST.values[[1]]

