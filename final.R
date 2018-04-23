park=read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"),header=TRUE)
library(glmnet)
library(gbm)
library(randomForest)
attach(park)
Park=park[,c(-1,-5)] # delete subject and motor_UPDRS
Park=na.omit(Park)
X=model.matrix(total_UPDRS~.,Park)[,-1]
y=Park$total_UPDRS
n=dim(X)[1] # number of rows
p=dim(X)[2] # number of predictors/features
set.seed(1)
train.portion           = 0.5;     
m.samp                  = 100;
test.lasso.mse          = rep(0, m.samp)
train.lasso.mse         = rep(0, m.samp)
test.ridge.mse          = rep(0, m.samp)
train.ridge.mse         = rep(0, m.samp)
test.lasso.1se.mse      = rep(0, m.samp)
train.lasso.1se.mse     = rep(0, m.samp)
test.ridge.1se.mse      = rep(0, m.samp)
train.ridge.1se.mse     = rep(0, m.samp)
test.ls.mse             = rep(0, m.samp)
train.ls.mse            = rep(0, m.samp)
test.rf.mse             = rep(0, m.samp)
test.boost.mse1         = rep(0, m.samp)
test.regtree.mse        = rep(0, m.samp)
test.lasso.loocv.mse    = rep(0, m.samp)
test.lasso.1se.loocv.mse= rep(0, m.samp)
test.ridge.loocv.mse    = rep(0, m.samp)
test.ridge.1se.loocv.mse= rep(0, m.samp)
test.elnet.loocv.mse    = rep(0, m.samp)
test.elnet.mse          = rep(0, m.samp)
test.elnet.1se.loocv.mse= rep(0, m.samp)
test.elnet.1se.mse      = rep(0, m.samp)
test.lasso.AIC.mse      = rep(0, m.samp)
r2.rf                   = rep(0, m.samp)
r2.ls                   = rep(0, m.samp)
r2.lasso                = rep(0, m.samp)
r2.lasso.1se            = rep(0, m.samp)
r2.lasso.AIC            = rep(0, m.samp)
r2.ridge                = rep(0, m.samp)
r2.ridge.1se            = rep(0, m.samp)
r2.elnet                = rep(0, m.samp)
r2.elnet.1se            = rep(0, m.samp)

for (i in 1:m.samp){  
  train    =     sample(1:n, 3*p)
  test     =     (-train)
  X.train  =     X[train, ]
  y.train  =     y[train]
  X.test   =     X[test, ]
  y.test   =     y[test]
  # LS REGRESSION
  ls.fit           = glmnet(X.train, y.train, lambda = 0)
  y.hat.ls.test    = predict(ls.fit, newx = X.test) 
  test.ls.mse[i]   = mean((y.hat.ls.test - y.test)^2)
  r2.ls[i]         = sum((y.hat.ls.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  # LASSO REGRESSION
  lasso.mod        = glmnet(X.train, y.train)
  lasso.cv         = cv.glmnet(X.train, y.train)
  y.hat.lasso.test = predict(lasso.mod, s=lasso.cv$lambda.min, newx=X.test)
  test.lasso.mse[i]= mean((y.hat.lasso.test - y.test)^2)
  r2.lasso[i]      = sum((y.hat.lasso.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  tLL  <- lasso.mod$nulldev - deviance(lasso.mod)
  r    <- lasso.mod$df
  t    <- lasso.mod$nobs
  AICc <- -tLL+2*r+2*r*(r+1)/(t-r-1)
  lambda_aicc=lasso.mod$lambda[which.min(AICc)]
  y.hat.lasso.test=predict(lasso.mod, s=lambda_aicc, newx=X.test)
  test.lasso.AIC.mse[i]=mean((y.hat.lasso.test - y.test)^2)
  r2.lasso.AIC[i]=sum((y.hat.lasso.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  #1SE
  y.hat.lasso.test      = predict(lasso.mod,s=lasso.cv$lambda.1se,newx=X.test) 
  test.lasso.1se.mse[i] = mean((y.hat.lasso.test - y.test)^2)
  r2.lasso.1se[i]       = sum((y.hat.lasso.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  # Ridge REGRESSION, alpha = 0 is ridge
  ridge.mod        = glmnet(X.train, y.train, alpha = 0)
  ridge.cv         = cv.glmnet(X.train, y.train,alpha=0)
  y.hat.ridge.test = predict(ridge.mod, s=ridge.cv$lambda.min, newx=X.test)
  test.ridge.mse[i]= mean((y.hat.ridge.test - y.test)^2)
  r2.ridge[i]      = sum((y.hat.ridge.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  # 1SE
  y.hat.ridge.test     = predict(ridge.mod,s=ridge.cv$lambda.1se,newx=X.test)
  test.ridge.1se.mse[i]= mean((y.hat.ridge.test - y.test)^2)
  r2.ridge.1se[i]      = sum((y.hat.ridge.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  #Elnet, alpha=0.5
  elnet.mod        = glmnet(X.train, y.train, alpha = 0.5)
  elnet.cv         = cv.glmnet(X.train, y.train,alpha=0.5)
  y.hat.elnet.test = predict(elnet.mod,s=elnet.cv$lambda.min,newx=X.test)
  test.elnet.mse[i]= mean((y.hat.elnet.test - y.test)^2)
  r2.elnet[i]      = sum((y.hat.elnet.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  # 1SE
  y.hat.elnet.test     = predict(elnet.mod,s=elnet.cv$lambda.1se,newx=X.test)
  test.elnet.1se.mse[i]= mean((y.hat.elnet.test - y.test)^2)
  r2.elnet.1se[i]      = sum((y.hat.elnet.test-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  #random forest
  rf.mod=randomForest(total_UPDRS~.,data= Park,subset=train,importance=FALSE,ntree=100,mtry=p/3)
  y.hat.rf                              = predict(rf.mod, newdata=Park[test,])
  test.rf.mse[i]                        = mean( (y.hat.rf-y.test)^2 )
  
  #r2.rf[i]=1-sum((y.hat.rf-y.test)^2)/sum((mean(y.test)-y.test)^2)
  r2.rf[i]=sum((y.hat.rf-mean(y.test))^2)/sum((mean(y.test)-y.test)^2)
  
  #boosting
  boost.mod=gbm(total_UPDRS~.,data=Park[train,],distribution="gaussian",n.trees=5000,interaction.depth=3,verbose=F)  
  yhat.boost=predict(boost.mod, newdata=Park[-train,], n.trees=5000)
  test.boost.mse1[i]=mean( (yhat.boost-y.test)^2 )
}
par(mfrow=c(1,1))

test.mse<-data.frame(test.boost.mse1,test.rf.mse,test.ridge.mse,test.ridge.1se.mse,test.lasso.mse,test.lasso.1se.mse,test.elnet.mse,test.elnet.1se.mse,test.ls.mse,test.lasso.AIC.mse)
colnames(test.mse)<-c('boosting',"RF","ridge","ridge.1se","lasso","lasso.1se","elnet","elnet.1se","ls","lassoAIC")
boxplot(test.mse,main='nlearn=3*p',ylab='test MSE')

plot(lasso.cv,main='nlearn=3*p',ylab='test MSE')
plot(ridge.cv,main='nlearn=3*p',ylab='test MSE')
plot(log(lasso.mod$lambda),AICc,xlab='log(Lambda)',main='Lasso AICc nlearn=3*p')

par(mfrow=c(1,3))
barplot(rf.mod$importance[,1]/sum(rf.mod$importance),horiz=TRUE,las=1,main='Random Forest nlearn=3*p')
barplot(abs(coef(lasso.mod,s=lasso.cv$lambda.min)[-1,1])*apply(X,2,sd)/sum(abs(coef(lasso.mod,s=lasso.cv$lambda.min)[-1,1])*apply(X,2,sd)),horiz=TRUE,las=1,main='Lasso nlearn=3*p')
barplot(abs(coef(ridge.mod,s=ridge.cv$lambda.min)[-1,1])*apply(X,2,sd)/sum(abs(coef(ridge.mod,s=ridge.cv$lambda.min)[-1,1])*apply(X,2,sd)),horiz=TRUE,las=1,main='Ridge nlearn=3*p')

# test.r2<-data.frame(r2.rf,r2.ridge,r2.ridge.1se,r2.lasso,r2.lasso.1se,r2.lasso.AIC,r2.elnet,r2.elnet.1se,r2.ls)
# colnames(test.r2)<-c("RF","ridge","ridge.1se","lasso","lasso.1se","lassoAIC","elnet","elnet.1se","ls")
# boxplot(test.r2,main='R2 nlearn=3*p',ylab='test R2')


# tree.mod=tree(total_UPDRS~., Park, subset=train)
# summary(tree.mod)
# plot(tree.mod)
# text(tree.mod,pretty=0)
# cv.treemod=cv.tree(tree.mod)
# plot(cv.treemod$size,cv.treemod$dev,type='b')
# prune.treemod=prune.tree(tree.mod, best=cv.treemod$size[which.min(cv.treemod$dev)])
# plot(prune.treemod)
# text(prune.treemod,pretty=0)
# yhat_tree=predict(tree.mod, newdata=Park[-train,])
# plot(yhat_tree,y.test)
# abline(0,1,col='red')
# 
# 
# for (i in 1:m.samp){ 
#   train    =     sample(1:n, 3*p)
#   test     =     (-train)
#   X.train  =     X[train, ]
#   y.train  =     y[train]
#   X.test   =     X[test, ]
#   y.test   =     y[test]
#   tree.mod=tree(total_UPDRS~., Park, subset=train)
#   cv.treemod=cv.tree(tree.mod)
#   prune.treemod=prune.tree(tree.mod, best=cv.treemod$size[which.min(cv.treemod$dev)])
#   yhat_tree=predict(prune.treemod, newdata=Park[-train,])
#   test.regtree.mse[i]=mean( (yhat_tree-y.test)^2 )
# }