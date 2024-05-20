install.packages("reticulate")
install.packages("tm")
install.packages("SnowballC")
install.packages("dplyr")
install.packages("ggplot")


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

## we will now make comments a dataframe before creating Corpus
comments_df <- data.frame(Comments = comments)
print(comments_df)

## Sentiment Analysis Package

install.packages("SentimentAnalysis")
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

## create tf idf document term matrix

tfidf <- DocumentTermMatrix(corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
inspect(tfidf[1:20, ])


#split comments based on the sentiment
pos_comments <- comments_df[comments_df$Sentiment > 0, "Comments"]
neg_comments <- comments_df[comments_df$Sentiment < 0, "Comments"]
neut_comments <- comments_df[comments_df$Sentiment == 0, "Comments"]

#Create corpus for every sentiment
corpus_positive <- Corpus(VectorSource(pos_comments))
corpus_negative <- Corpus(VectorSource(neg_comments))
corpus_neutral <- Corpus(VectorSource(neut_comments))

inspect(corpus_negative)

##now we shall preprocess each corpus for each sentiment, since we are applying the same steps
##lets just use a function

preprocess_corpus <- function(corpus) {
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeWords, stopwords("en"))
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, removeWords, custom_stopwords)
  return(corpus)
}

corpus_positive <- preprocess_corpus(corpus_positive)
corpus_negative <- preprocess_corpus(corpus_negative)
corpus_neutral <- preprocess_corpus(corpus_neutral)

inspect(corpus_negative)

# Create DTMs for each sentiment
dtm_positive <- DocumentTermMatrix(corpus_positive, control = list(bounds = list(global = c(2, Inf))))
dtm_negative <- DocumentTermMatrix(corpus_negative, control = list(bounds = list(global = c(2, Inf))))
dtm_neutral <- DocumentTermMatrix(corpus_neutral, control = list(bounds = list(global = c(2, Inf))))
inspect(dtm_negative)

# Create TF-IDF DTMs for each sentiment
tfidf_positive <- DocumentTermMatrix(corpus_positive, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
tfidf_negative <- DocumentTermMatrix(corpus_negative, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
tfidf_neutral <- DocumentTermMatrix(corpus_neutral, control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
inspect(tfidf_neutral)

#converting positive matrix to dataframe for analysis

DocumentTermDataFrame <- as.data.frame(as.matrix(dtm_positive))

#get list of all terms
names(DocumentTermDataFrame)

#get top words in corpus
WordFreq <- colSums(DocumentTermDataFrame)
View(WordFreq)

#convert it to dataframe
WordFreq_df <- data.frame(Term = names(WordFreq), Frequency = WordFreq, row.names = NULL)
View(WordFreq_df)

library(dplyr)

## view the top positive words

top_words_all <-  WordFreq_df %>%
  top_n(20, Frequency) %>%
  arrange(desc(Frequency))

View(top_words_all)

library(ggplot2)


#Plot it
ggplot(top_words_all, aes(x = reorder(Term, Frequency), y = Frequency)) +
  geom_bar(stat = "identity") +
  labs(x = "Word", y = "Frequency", title = "Top Words All") +
  coord_flip()

#6: Latent Semantic Analysis (LSA) ----
#Creates a Feature/Context space, where similar words have similar representation
#You can see how many Context/Feature each word could have

install.packages("lsa")
library("lsa")

corpus_tdm<-TermDocumentMatrix(corpus, 
                             control = list(bounds = list(global = c(2, Inf))))
inspect(corpus_tdm)

#we shall automatically generate the number of dimension

lsaSpace<-lsa(corpus_tdm,dims=dimcalc_share())
lsaSpace

#the matrix represents a set of 50 terms in a 20-dimensional space.

#convert to matrix
#what row and columns are representing?
TermDocDataMatrix <- as.textmatrix(lsaSpace)
TermDocDataMatrix

##TermDocDataMatrix shows that each row corresponds to a term (word) and each column 
##corresponds to a document. The values in the matrix represent the frequency of
##each term in each document.

#get term/feature 
termSpace<-lsaSpace$tk
lables=rownames(termSpace)
termSpace

#get document/feature 
docSpace<-lsaSpace$dk
lablesdoc=rownames(docSpace) #98 documents are present

#get feature space
contextSpace<-lsaSpace$sk
contextSpace

library(ggrepel)

#plotting the terms over 2 dimention
ggplot(as.data.frame(termSpace), aes(x = termSpace[,1], y=termSpace[,2], 
                                     label=lables)) +
  geom_point() +
  geom_jitter()+
  geom_text_repel(max.overlaps = 15) +
  labs(x = "Context1", y = "Context2", title = "Terms in Space")

