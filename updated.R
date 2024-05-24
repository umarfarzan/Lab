
getwd()

setwd('C:/Users/1/Downloads/411_Lab4/')


library(SnowballC)
options(max.print = 1000)


library(reticulate)

virtualenv_create(envname = "yt_env")

virtualenv_install(envname = "yt_env", packages = "google-api-python-client")

use_virtualenv("yt_env", required = TRUE)

source_python("ytcom.py")

api_key <- "AIzaSyA3HK0XOK2b-Hh56FKdxm9_bh9C4BnQ0to"
video_id <- "0X0Jm8QValY"

comments <- get_youtube_comments(api_key, video_id, max_results = 500)

print(comments)


comments_df <- data.frame(Comments = comments)
print(comments_df)

## Sentiment Analysis Package

library(SentimentAnalysis)

#REMOVE HTML TAGS

remove_html_tags <- function(text) {
  gsub("<.*?>", "", text)
}

#apply function to comments

comments_df$Comments <- sapply(comments_df$Comments, remove_html_tags)
print(comments_df)

# Remove rows with NA values
comments_df <- na.omit(comments_df)

#sentiment analysis results
?analyzeSentiment

results <- analyzeSentiment(comments_df$Comments)

print(results)

#add sentiment results to dataframe

comments_df$Sentiment <- results$SentimentQDAP

print(comments_df)

## creating a corpus
library(tm)
corpus <- Corpus(VectorSource(comments_df$Comments))
corpus

#do the preprocessing

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords("en"))
corpus <- tm_map(corpus, stripWhitespace)
print(corpus)

# Remove custom stopwords
custom_stopwords <- c("chars", "su", "--", "”", "“")
corpus <- tm_map(corpus, removeWords, custom_stopwords)

inspect(corpus)

#create a document term matrix

dtm <- DocumentTermMatrix(corpus, control = list(bounds = list(global = c(2, Inf))))
inspect(dtm[1:20, ])


#due to strong preprocessing, some of the rows in the dataframe have all 0s in the row
#remove them using slam package

library(slam)
row_sums <- slam::row_sums(dtm, na.rm = TRUE)
dtm <- dtm[row_sums > 0, ]

dtm
inspect(dtm[1:20, ])


library("topicmodels")

library("ldatuning")

#FINDING TOPICS NUMBER

result <- FindTopicsNumber(
  dtm,
  topics = seq(from = 2, to = 50, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(result)


#LDA FIT TRYING WITH DIFFERENT NUMBER OF TOPICS ACCORDING TO PREVIOUS PLOT RESULTS
?LDA
ldaResult <-LDA(dtm, 8, method="Gibbs", control=list(nstart=8, seed = list(1,2,3,4,5,6,7,8), best=TRUE,  iter = 20))
ldaResult <-LDA(dtm, 6, method="Gibbs", control=list(nstart=6, seed = list(1,2,3,4,5,6), best=TRUE,  iter = 20))


ldaResult.terms <- as.matrix(terms(ldaResult, 10))
ldaResult.terms

inspect(ldaResult.terms)

install.packages("LDAvis")
library("LDAvis")

#FUNCTION FOR VISUALIZING THE LDA RESULTS

topicmodels2LDAvis <- function(x, ...){
  post <- topicmodels::posterior(x)
  if (ncol(post[["topics"]]) < 3) stop("The model must contain > 2 topics")
  mat <- x@wordassignments
  LDAvis::createJSON(
    phi = post[["terms"]], 
    theta = post[["topics"]],
    vocab = colnames(post[["terms"]]),
    doc.length = slam::row_sums(mat, na.rm = TRUE),
    term.frequency = slam::col_sums(mat, na.rm = TRUE)
  )
}

install.packages('servr')
serVis(topicmodels2LDAvis(ldaResult))


#KILLING SERVERS
servr::daemon_stop(1)
servr::daemon_stop(2)
servr::daemon_stop(3)
