##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

# Necessary Packages & Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Download, unzip and mutate our dataset into a usable state.
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create test and train set for training and then testing our algorithm
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Function to compare our predictions against the test set, ignoring NA.
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = T))
}

# Mean and Median Rating
mean(edx$rating)
median(edx$rating)

# Explore the Genres column
edx %>% filter(str_detect(genres, 'Romance')) %>% select(rating) %>% colMeans()

edx %>% filter(str_detect(genres, '^Romance$')) %>% select(rating) %>% colMeans()

edx %>% filter(str_detect(genres, '^Drama$')) %>% select(rating) %>% colMeans()
edx %>% filter(str_detect(genres, 'Drama')) %>% select(rating) %>% colMeans()

# Another form of Root Mean Square Error
sqrt(mean((test_set$rating - mean(train_set$rating))^2))

# Generate betas from the movieId column
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu))

# Generate predictions using b_m
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  summarize(pred = mu + b_m) %>%
  .$pred

RMSE(test_set$rating, predicted_ratings)

# Graph of User Average Ratings
train_set[1:10000] %>% 
  group_by(userId) %>% 
  summarize(avg = mean(rating), id = userId) %>% 
  ggplot(aes(x=id, y=avg, colour='smooth')) + 
  geom_point() + xlab('User') + 
  ylab('Average Rating') + 
  ggtitle('Subset of User\'s Average Rating', subtitle = 'Sized by Number of Reviews') + 
  theme_light() + 
  theme(legend.position = 'none')

# Calculate effects from the user variable
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))

# Generate Ratings predictions using b_m and b_u (Beta-User)
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_m + b_u) %>%
  .$pred

RMSE(test_set$rating, predicted_ratings)

# Example of getting the Year
edx[1]$title
str_sub(edx[1]$title, -5, -2)

# Graph of Yearly Averages
train_set %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2))) %>% 
  group_by(year) %>% 
  summarize(year_avg = mean(rating)) %>% 
  ggplot(aes(year, year_avg)) + 
  geom_smooth() + 
  theme_light() + 
  xlab('Year') + 
  ylab('Average Rating') + 
  ggtitle('Average Rating across Years')

# Generate beta from Year variable
year_avgs <- train_set %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2))) %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%	
  group_by(year) %>% 
  summarize(b_y = mean(rating - mu - b_m - b_u))

# Generate predictions using b_m, b_u and b_y
predicted_ratings <- test_set %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year') %>%
  mutate(pred = mu + b_m + b_u + b_y) %>%
  .$pred

RMSE(test_set$rating, predicted_ratings)

# Generate beta from the genre variable (grouped)
genre_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  left_join(user_avgs, by='userId') %>% 
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_m - b_u))

# Generate predictions using b_m, b_u and b_g
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_m + b_u + b_g) %>%
  .$pred

RMSE(test_set$rating, predicted_ratings)

# Highest average ratings. Example of an effect with a small denominator.
train_set %>% group_by(movieId) %>% 
  summarize(avg = mean(rating), n = length(rating)) %>% 
  filter(avg==5)

# Graph of Yearly Averages with size
train_set %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2))) %>% 
  group_by(year) %>% 
  summarize(year_avg = mean(rating), n = length(rating)) %>% 
  ggplot(aes(year, year_avg, size=n, colour="smooth")) + 
  geom_point() + 
  theme_light() + 
  xlab('Year') + 
  ylab('Average Rating') + 
  ggtitle('Average Rating across Year', subtitle = 'Sized by Number of Reviews') + 
  theme(legend.position = 'none')

# Line 192 to 227
# Regenerate all of our previous betas, but with a lambda in the denominator
lambda <- 7
movie_avgs_reg <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_m = sum(rating - mu)/(n()+lambda))

user_avgs_reg <- train_set %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_m)/(n()+lambda))

genre_avgs_reg <- train_set %>% 
  left_join(movie_avgs_reg, by='movieId') %>% 
  left_join(user_avgs_reg, by='userId') %>% 
  group_by(genres) %>% 
  summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+lambda))

year_avgs_reg <- train_set %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2))) %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  left_join(user_avgs_reg, by='userId') %>%	
  left_join(genre_avgs_reg, by='genres') %>%	
  group_by(year) %>% 
  summarize(b_y = sum(rating - mu - b_m - b_u - b_g)/(n()+lambda))

predicted_ratings <- test_set %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2))) %>%
  left_join(movie_avgs_reg, by='movieId') %>%
  left_join(user_avgs_reg, by='userId') %>%
  left_join(genre_avgs_reg, by='genres') %>%
  left_join(year_avgs_reg, by='year') %>%
  mutate(pred = mu + b_m + b_u + b_g + b_y) %>%
  .$pred

RMSE(test_set$rating, predicted_ratings)


# Line 232 to 272
# Once again regenerate all our previous betas, but exploring different lambdas
lambdas <- c(1:10)

rmses <- sapply(lambdas, function(lambda){
  movie_avgs_reg <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - mu)/(n()+lambda))
  
  user_avgs_reg <- train_set %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_m)/(n()+lambda))
  
  genre_avgs_reg <- train_set %>% 
    left_join(movie_avgs_reg, by='movieId') %>% 
    left_join(user_avgs_reg, by='userId') %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+lambda))
  
  year_avgs_reg <- train_set %>% 
    mutate(year = as.numeric(str_sub(title,-5,-2))) %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%	
    left_join(genre_avgs_reg, by='genres') %>%	
    group_by(year) %>% 
    summarize(b_y = sum(rating - mu - b_m - b_u - b_g)/(n()+lambda))
  
  predicted_ratings <- test_set %>% 
    mutate(year = as.numeric(str_sub(title,-5,-2))) %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    left_join(year_avgs_reg, by='year') %>%
    mutate(pred = mu + b_m + b_u + b_g + b_y) %>%
    .$pred
  
  RMSE(test_set$rating, predicted_ratings)
})

# Discover the best lambda to use
lambdas[which.min(rmses)]
rmses[which.min(rmses)]

# Use our refined algorithm to generate betas from the full edx set and test against the validation set.
mu <- mean(edx$rating)
lambdas <- c(4:6)

rmses <- sapply(lambdas, function(lambda){
  movie_avgs_reg <- edx %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - mu)/(n()+lambda))
  
  user_avgs_reg <- edx %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_m)/(n()+lambda))
  
  genre_avgs_reg <- edx %>% 
    left_join(movie_avgs_reg, by='movieId') %>% 
    left_join(user_avgs_reg, by='userId') %>% 
    group_by(genres) %>% 
    summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+lambda))
  
  year_avgs_reg <- edx %>% 
    mutate(year = as.numeric(str_sub(title,-5,-2))) %>% 
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%	
    left_join(genre_avgs_reg, by='genres') %>%	
    group_by(year) %>% 
    summarize(b_y = sum(rating - mu - b_m - b_u - b_g)/(n()+lambda))
  
  predicted_ratings <- validation %>% 
    mutate(year = as.numeric(str_sub(title,-5,-2))) %>%
    left_join(movie_avgs_reg, by='movieId') %>%
    left_join(user_avgs_reg, by='userId') %>%
    left_join(genre_avgs_reg, by='genres') %>%
    left_join(year_avgs_reg, by='year') %>%
    mutate(pred = mu + b_m + b_u + b_g + b_y) %>%
    .$pred
  
  RMSE(validation$rating, predicted_ratings)
})

lambdas[which.min(rmses)]
#Final RMSE
rmses[which.min(rmses)]
