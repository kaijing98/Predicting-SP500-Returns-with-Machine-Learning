# 4. Text analytics
install.packages("dplyr")
install.packages("tidytext")
install.packages("tidyr")
install.packages("textdata")
library(dplyr)
library(tidytext)
library(tidyr)
library(textdata)
data(stop_words)

# Train
sti1 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Jan.txt", encoding ="UTF-8"))
sti2 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Feb.txt", encoding ="UTF-8"))
sti3 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Mar.txt", encoding ="UTF-8"))
sti4 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Apr.txt", encoding ="UTF-8"))
sti5 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/May.txt", encoding ="UTF-8"))
sti6 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Jun.txt", encoding ="UTF-8"))
sti7 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Jul.txt", encoding ="UTF-8"))
sti8 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Aug.txt", encoding ="UTF-8"))
sti9 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Sep.txt", encoding ="UTF-8"))
sti10 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Oct.txt", encoding ="UTF-8"))
sti11 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Nov.txt", encoding ="UTF-8"))
sti12 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Dec.txt", encoding ="UTF-8"))

sti1 <- data_frame(text = sti1)
sti1.t <- sti1 %>% unnest_tokens(word, text)
sti1.t <- sti1.t %>% anti_join(stop_words)

sti2 <- data_frame(text = sti2)
sti2.t <- sti2 %>% unnest_tokens(word, text)
sti2.t <- sti2.t %>% anti_join(stop_words)

sti3 <- data_frame(text = sti3)
sti3.t <- sti3 %>% unnest_tokens(word, text)
sti3.t <- sti3.t %>% anti_join(stop_words)

sti4 <- data_frame(text = sti4)
sti4.t <- sti4 %>% unnest_tokens(word, text)
sti4.t <- sti4.t %>% anti_join(stop_words)

sti5 <- data_frame(text = sti5)
sti5.t <- sti5 %>% unnest_tokens(word, text)
sti5.t <- sti5.t %>% anti_join(stop_words)

sti6 <- data_frame(text = sti6)
sti6.t <- sti6 %>% unnest_tokens(word, text)
sti6.t <- sti6.t %>% anti_join(stop_words)

sti7 <- data_frame(text = sti7)
sti7.t <- sti7 %>% unnest_tokens(word, text)
sti7.t <- sti7.t %>% anti_join(stop_words)

sti8 <- data_frame(text = sti8)
sti8.t <- sti8 %>% unnest_tokens(word, text)
sti8.t <- sti8.t %>% anti_join(stop_words)

sti9 <- data_frame(text = sti9)
sti9.t <- sti9 %>% unnest_tokens(word, text)
sti9.t <- sti9.t %>% anti_join(stop_words)

sti10 <- data_frame(text = sti10)
sti10.t <- sti10 %>% unnest_tokens(word, text)
sti10.t <- sti10.t %>% anti_join(stop_words)

sti11 <- data_frame(text = sti11)
sti11.t <- sti11 %>% unnest_tokens(word, text)
sti11.t <- sti11.t %>% anti_join(stop_words)

sti12 <- data_frame(text = sti12)
sti12.t <- sti12 %>% unnest_tokens(word, text)
sti12.t <- sti12.t %>% anti_join(stop_words)

sentiments
get_sentiments("bing")
## Neg or Pos for each word
get_sentiments("afinn")
## Scoring from - 5 to +5 for each word

# Most common positive and negative words by Bing lexicon 
# Spread() in package tidyr to list Neg and Pos sentiments in another columns
sti1.sen <- sti1.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti2.sen <- sti2.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti3.sen <- sti3.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti4.sen <- sti4.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti5.sen <- sti5.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti6.sen <- sti6.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti7.sen <- sti7.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti8.sen <- sti8.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti9.sen <- sti9.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti10.sen <- sti10.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti11.sen <- sti11.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti12.sen <- sti12.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

# Sentiment by Afinn score -5 to 5 per word 

sti1.sen.af <- sti1.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti2.sen.af <- sti2.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti3.sen.af <- sti3.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti4.sen.af <- sti4.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti5.sen.af <- sti5.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti6.sen.af <- sti6.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti7.sen.af <- sti7.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti8.sen.af <- sti8.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti9.sen.af <- sti9.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti10.sen.af <- sti10.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti11.sen.af <- sti11.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti12.sen.af <- sti12.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))

# Overall Sentiment Across the 3 Rallys
overall <- data.frame(Sentiment.Bing = c(0,0,0,0,0,0,0,0,0,0,0,0), Sentiment.Afinn = c(0,0,0,0,0,0,0,0,0,0,0,0), row.names = c("sti1","sti2","sti3","sti4","sti5","sti6","sti7","sti8","sti9","sti10","sti11","sti12"))

overall$Sentiment.Bing[1] <- sti1.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[2] <- sti2.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[3] <- sti3.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[4] <- sti4.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[5] <- sti5.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[6] <- sti6.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[7] <- sti7.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[8] <- sti8.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[9] <- sti9.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[10] <- sti10.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[11] <- sti11.sen %>% summarise(sum(sentiment))
overall$Sentiment.Bing[12] <- sti12.sen %>% summarise(sum(sentiment))

overall$Sentiment.Afinn[1] <- sti1.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[2] <- sti2.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[3] <- sti3.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[4] <- sti4.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[5] <- sti5.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[6] <- sti6.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[7] <- sti7.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[8] <- sti8.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[9] <- sti9.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[10] <- sti10.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[11] <- sti11.sen.af %>% summarise(sum(value*n))
overall$Sentiment.Afinn[12] <- sti12.sen.af %>% summarise(sum(value*n))

overall

# Test
sti13 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Jan19.txt", encoding ="UTF-8"))
sti14 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Feb19.txt", encoding ="UTF-8"))
sti15 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Mar19.txt", encoding ="UTF-8"))
sti16 <- readLines(con <-file("D:/NUS Files JX/Y5S1/EC4308/Project/Apr19.txt", encoding ="UTF-8"))

sti13 <- data_frame(text = sti13)
sti13.t <- sti13 %>% unnest_tokens(word, text)
sti13.t <- sti13.t %>% anti_join(stop_words)

sti14 <- data_frame(text = sti14)
sti14.t <- sti14 %>% unnest_tokens(word, text)
sti14.t <- sti14.t %>% anti_join(stop_words)

sti15 <- data_frame(text = sti15)
sti15.t <- sti15 %>% unnest_tokens(word, text)
sti15.t <- sti15.t %>% anti_join(stop_words)

sti16 <- data_frame(text = sti16)
sti16.t <- sti16 %>% unnest_tokens(word, text)
sti16.t <- sti16.t %>% anti_join(stop_words)

sentiments
get_sentiments("bing")
## Neg or Pos for each word
get_sentiments("afinn")
## Scoring from - 5 to +5 for each word

sti13.sen <- sti13.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti14.sen <- sti14.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti15.sen <- sti15.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti16.sen <- sti16.t %>% inner_join(get_sentiments("bing")) %>% count(word, sentiment) %>% 
  spread(sentiment, n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(desc(sentiment))

sti13.sen.af <- sti13.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti14.sen.af <- sti14.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti15.sen.af <- sti15.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))
sti16.sen.af <- sti16.t %>% inner_join(get_sentiments("afinn")) %>% count(word, value) %>% arrange(desc(value))

# Overall Sentiment Across the 3 Rallys
overall2 <- data.frame(Sentiment.Bing = c(0,0,0,0), Sentiment.Afinn = c(0,0,0,0), row.names = c("sti13","sti14","sti15","sti16"))

overall2$Sentiment.Bing[1] <- sti13.sen %>% summarise(sum(sentiment))
overall2$Sentiment.Bing[2] <- sti14.sen %>% summarise(sum(sentiment))
overall2$Sentiment.Bing[3] <- sti15.sen %>% summarise(sum(sentiment))
overall2$Sentiment.Bing[4] <- sti16.sen %>% summarise(sum(sentiment))

overall2$Sentiment.Afinn[1] <- sti13.sen.af %>% summarise(sum(value*n))
overall2$Sentiment.Afinn[2] <- sti14.sen.af %>% summarise(sum(value*n))
overall2$Sentiment.Afinn[3] <- sti15.sen.af %>% summarise(sum(value*n))
overall2$Sentiment.Afinn[4] <- sti16.sen.af %>% summarise(sum(value*n))

overall2