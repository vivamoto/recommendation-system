

################################
# Rascunho do capstone
################################

# Report (40 points possible)
Your submission for this project is three files:
  
  1. Your report in PDF format
  1. Your report in Rmd format
  1. A script in R format that generates your predicted movie ratings and RMSE score

The report documents the analysis and presents the findings, along with 
supporting statistics and figures. 

The report must include the RMSE generated. 

40 points: The report includes all required sections, 
is easy to follow with good supporting detail throughout, 
and is insightful and innovative. 

# RMSE (25 points)

Provide the appropriate score given the reported RMSE. *Please be sure not to use the validation set for training or regularization - you may wish to create an additional partition of training and test sets from the provided edx dataset to experiment with multiple parameters or use cross-validation.*
  
  
  10 points: 0.86550 <= RMSE <= 0.89999  
  15 points: 0.86500 <= RMSE <= 0.86549  *
  20 points: 0.86490 <= RMSE <= 0.86499  
  25 points: RMSE <= 0.8649
                     0.8648177
                     0.8641362
                     0.8646783 

|Method         |      RMSE|
|:--------------|---------:|
|Mean           | 1.0603313|
|Mean + bi      | 0.9439087|
|Mean + bi + bu | 0.8653488|
                    86481

|Method                |      RMSE|
|:---------------------|---------:|
|Mean                  | 1.0600537|
|Mean + bi             | 0.9429615|
|Mean + bi + bu        | 0.8646843|
|Regularized bi and bu | 0.8641362|                    
  
  


# Process genres
foo <- edx[1:5,]
str(foo$genres)
lev <- as.factor(edx$genres)
str(lev)
levels(lev)
nlevels(lev)
as.numeric(lev)%>% head()
length(lev)
head(edx)
paste(foo$genres[1],foo$genres[2], sep="|")
# Create a vector of unique genres
res <- tibble(genre = parse_guess(str_split(foo$genres[1], "\\|", simplify = TRUE)))
for(x in 2:length(foo$genres)){
  res <- bind_rows(res, tibble(genre = parse_guess(str_split(foo$genres[1], "\\|", simplify = TRUE)))) %>% distinct()
}
res


# Apply to edx
# Create a vector of unique genres
# (this code takes long time to process)
res <- tibble(genre = parse_guess(str_split(edx$genres[1], "\\|", simplify = TRUE)))
for(x in 2:length(edx$genres)){
  res <- bind_rows(res, tibble(genre = parse_guess(str_split(edx$genres[1], "\\|", simplify = TRUE)))) %>% distinct()
}
res

res <- edx$genres[1]
for(x in 2:length(edx$genres)){
  res <- paste(res,edx$genres[x], sep = "|")
}
res1 <- tibble(genre = parse_guess(str_split(res, "\\|", simplify=TRUE))) %>% distinct()
rm(res)

res2 <- distinct(res1)
str(res1)
library(purrr)
map(res1, function(x) x[[1]])

sapply(2:length(foo$genres), jl)

jl <- function(x) {paste(x,)}
sapply(foo$genres, function(x) )

bind_rows(genres_list[,1],genres_list[,1])

foo$genres[2]
str(genres_list)
str_subset(foo$genres, "|")

#----------------
# Profiling: check processing speed
#----------------
if(!require(profvis)) install.packages("profvis")
profvis({
  data(diamonds, package = "ggplot2")
  
  plot(price ~ carat, data = diamonds)
  m <- lm(price ~ carat, data = diamonds)
  abline(m, col = "red")
})

#------------------
# estudo de matrizes
#------------------
x <- 1:15
x
as.matrix(x)[1:3,1:3]
str(x)
class(x)

#-----------------
#
#----------------
# 2. Include movie effect (bi)
# bi is the movie effect (bias) for movie i.
# y_hat = mu + bi
bi <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
bi

# Predict the rating with mean + bi
y_hat_bi <- mu + test_set %>% 
  left_join(bi, by = "movieId") %>% 
  pull(b_i)

# Calculate the RMSE
result <- bind_rows(result, 
                    tibble(Method = "Mean + bi", 
                           RMSE = RMSE(test_set$rating, y_hat_bi)))

mean((y_hat_bi - test_set$rating)^2)  %>% as_tibble

sqrt(mean((true_ratings - predicted_ratings)^2))
tibble(y=y_hat_bi, x=test_set$movieId) %>% filter(is.na(y)) 
tibble(y=bi$b_i) %>% filter(is.na(.)) %>% count()

test_set[is.na(y_hat_bi)]
str(y_hat_bi)

#================================
# Linear Model - Testes diversos
#================================
# 1. Predict the same rating for all movies.
# The initial prediction is the mean of the ratings (mu).
# y_hat = mu
mu_edx <- mean(edx$rating)

# Calculate the RMSE
result_edx <- tibble(Method = "Mean EDX", RMSE = RMSE(test_set$rating, mu_edx))

# 2. Include movie effect (bi)
# bi is the movie effect (bias) for movie i.
# y_hat = mu + bi
bi_edx <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
bi_edx
qplot(b_i, data = bi_edx, bins = 10, color = I("black"))

# Predict the rating with mean + bi
y_hat_bi_edx <- mu + validation %>% 
  left_join(bi_edx, by = "movieId") %>% 
#  filter(!is.na(b_i)) %>%
  .$b_i
#length(y_hat_bi)
#length(test_set$rating)
#nrow(bi)
#head(bi)
sum(is.na(y_hat_bi_edx))
#RMSE(test_set$rating, y_hat_bi)
RMSE(validation$rating,mu + edx %>% left_join(bi, by = "movieId") %>% filter(!is.na(b_i)) %>%.$b_i)
#900007*9

# Calculate the RMSE

result_edx <- bind_rows(result_edx, 
                    tibble(Method = "Mean + bi EDX", 
                           RMSE = RMSE(validation$rating, y_hat_bi_edx)))

# Show the RMSE improvement
result_edx %>% knitr::kable()

# 3. Include user effect (bu)
# bu is the user effect (bias) for user u.
# y_hat = mu + bi + bu
bu_edx <- edx %>% 
  left_join(bi_edx, by = 'movieId') %>%
  filter(!is.na(b_i)) %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict the rating with mean + bi + bu
y_hat_bi_bu_edx <- validation %>% 
  left_join(bi_edx, by='movieId') %>%
  left_join(bu_edx, by='userId') %>%
#  filter(!is.na(b_i), !is.na(b_u)) %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred


result_edx <- bind_rows(result_edx, 
                        tibble(Method = "Mean + bi + bu", 
                           RMSE = RMSE(validation$rating, y_hat_bi_bu_edx)))

# Show the RMSE improvement
result_edx %>% knitr::kable()

edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")




rmses <- sapply(lambdas, regularization,dataset=train_set)
    
regularization <- function(l, dataset){
  
  b_i <- dataset %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
}

rmses

?sapply

x_1 <- 10
x <- 25
tst <- function(x){
  print(x_1)
  print(x)
  x1 <- 15
  x_2 <- 100
  return(c(x+5,x*10))
}
tst(5)
x_2
xx


#==============================
# Estudo do Recommenderlab
#==============================
package.install("recommenderlab")
install.packages("recommenderlab")

# Mostra a lista de funções disponíveis no pacote
recommenderRegistry$get_entries(dataType = "realRatingMatrix")


train_s <- as(train_set,"realRatingMatrix")
train_s
test_s <- as(test_set,"realRatingMatrix")
test_s

getRatingMatrix(test_s) %>% head()
train_s_m <- normalize(train_s)

hist(getRatings(train_s), breaks=100)
hist(getRatings(normalize(train_s)), breaks=100)
hist(getRatings(normalize(train_s, method="Z-score")), breaks=100)
hist(rowCounts(train_s), breaks=50)
hist(colMeans(train_s), breaks=20)

rec <- Recommender(train_s[1:1000], method = "POPULAR")
rec
names(getModel(rec))

y_hat_pop <- recommenderlab::predict(rec, train_s[1001:1010], n=5)
as(y_hat_pop,"list")
recom <- recommenderlab::predict(rec, train_s[1001:1002], type="ratings")

as(recom, "matrix")[,1:10]

#=====================
data(Jester5k)
Jester5k

as(Jester5k, "matrix")[1:5,1:5]
as(train_s,  "matrix")[1:5,1:5]

e <- evaluationScheme(Jester5k[1:1000], 
                      method="split", 
                      train=0.9,
                      given=15, 
                      goodRating=5)
e

# We create two recommenders (user-based and item-based 
# collaborative filtering) using the training data.
r1 <- Recommender(getData(e, "train"), "UBCF")
r1
r2 <- Recommender(getData(e, "train"), "IBCF")
r2

# Next, we compute predicted ratings for the known part 
# of the test data (15 items for each user) using the 
# two algorithms.

p1 <- predict(r1, getData(e, "known"), type="ratings")
p1
p2 <- predict(r2, getData(e, "known"), type="ratings")
p2

# Finally, we can calculate the error between the prediction 
# and the unknown part of the test data.
error <- rbind(
   UBCF = calcPredictionAccuracy(p1, getData(e, "unknown")),
   IBCF = calcPredictionAccuracy(p2, getData(e, "unknown"))
   )
error


#===============
# prepara edx e validation para recommenderlab
#===============

x <- train_set[1:10,]
x

x1 <- train_set[1:100,] %>% select(movieId, userId,rating) %>% spread(movieId,rating)
x1

train_s <- train_set %>% select(movieId, userId,rating) %>% 
                    spread(movieId,rating)


head(train_s)


qplot(pca$rotation[,1],pca$rotation[,2], xlab = "PC1", ylab="PC2")
 data.frame(PC1 = pca$rotation[,1],
           PC2 = pca$rotation[,2],
           tit = as.factor(dimnames(pca$rotation)[[1]])) %>%  
  ggplot(aes(PC1, PC2, label = tit)) +
  geom_point() +
  geom_text_repel() 
  
#-----------------------
# recosystem
r = Reco()
set.seed(123, sample.kind = "Rounding") # This is a randomized algorithm
r$train(data_file(train_set), opts = list(dim = 10, nmf = TRUE))

## Write P and Q matrices to files
P_file = out_file(tempfile())
Q_file = out_file(tempfile())
r$output(P_file, Q_file)
head(read.table(P_file@dest, header = FALSE, sep = " "))
head(read.table(Q_file@dest, header = FALSE, sep = " "))
## Skip P and only export Q
r$output(out_nothing(), Q_file)



#-----------------------
library(recosystem)
set.seed(123, sample.kind = "Rounding") # This is a randomized algorithm
train_data <-  with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_data  <-  with(test_set,  data_memory(user_index = userId, item_index = movieId, rating = rating))
r <-  recosystem::Reco()
opts <-  r$tune(train_data, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                     costp_l2 = c(0.01, 0.1), 
                                     costq_l2 = c(0.01, 0.1),
                                     nthread  = 4, niter = 10))
opts
r$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))

## Write predictions to file
pred_file <-  tempfile()
r$predict(test_data, out_file(pred_file))

print(scan(pred_file, n = 10)) 

## Or, directly return an R vector
pred_rvec <-  r$predict(test_data, out_memory())
head(pred_rvec, 10)

library(stringr)
train_set %>% filter(str_detect(title, "Back to the Future")) %>% head()



#------------------
#cria confusion matrix
#------------------

roundRating <- function(x) {
  y <- round(x,1)
  ifelse(y >= 4.8, 5, ifelse(y >= 4.3, 4.5,
        ifelse(y >= 3.8, 4, ifelse(y >= 3.3, 3.5,
               ifelse(y >= 2.8, 3, ifelse(y >= 2.3, 2.5,
                      ifelse(y >= 1.8, 2, ifelse(y >= 1.3, 1.5,
                            ifelse(y >= 0.8, 1, 0.5)))))))))
}

str(tibble(x = seq(0,7,.1), rating = as.factor(roundRating(x))))
confusionMatrix(as.factor(roundRating(y_hat_reg)), as.factor(test_set$rating))
confusionMatrix(as.factor(roundRating(y_hat_recon)), as.factor(test_set$rating))


head(y_hat_reg)
head(as.factor(roundRating(y_hat_reg)))
levels(test_set$rating)  

#------------------
# Random prediction
#------------------

set.seed(4321, sample.kind = "Rounding")
p <- function(x) mean(train_set$rating == x)

rating <- seq(0.5,5,0.5)

prob <- sapply(rating, p)
#-----
p <- function(x,y) mean(y == x)
rating <- seq(0.5,5,0.5)

B <- 1000
M <- replicate(B, {
  s <- sample(train_set$rating, 100)
  prob <- sapply(rating, p1, y= s)
})
M
prob <- sapply(1:nrow(M), function(x) mean(M[x,]))
sapply(1:ncol(M), function(x) sum(M[,x]))
sum(sapply(1:nrow(M), function(x) mean(M[x,])))


p1 <- function(x) mean(M == x)

y_hat_random <- sample(rating, size = nrow(test_set), replace = TRUE, prob = prob)
RMSE(test_set$rating, y_hat_random)
MSE(test_set$rating, y_hat_random)
MAE(test_set$rating, y_hat_random)


seq(0.5,5,0.5)


#====================
edx1 <- edx[1:100,]

edx1 %>% 
  group_by(genres) %>%
  summarize(avg = mean(rating), 
            se  = sd(rating)/sqrt(n()),
            min = avg - 2*se, 
            max = avg + 2*se,
            n = n()) %>%
  filter(n >= 1) %>%
  mutate(genres = reorder(genres, avg)) %>%
  mutate(recode())
str_de
  