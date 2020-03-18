Customer reviews with Natural Language Processing
================

We have the reviews of a restaurant. We will use machine learning NLP
algorithm to predict if the review is positive or negative.

``` r
# Natural Language Processing

# Importing the dataset
dataset_original = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)
```

Dataset is presented in the tsv file format. It is a tab-separated
values format which is very similar to csv (comma-separated values)
format. This format is considered to be better then csv for NLP because
commas are very likely to be a part of a sentence and csv file will
recognize them as separators. And tabs are not likely to be the part of
the sentence. So mind this and always use tsv format.

Here we have only two columns: Review and Liked. Liked is 1 for positive
comments and 0 for negative

``` r
# Cleaning the texts
#install.packages('tm')
#install.packages('SnowballC')
library(tm)
```

    ## Warning: package 'tm' was built under R version 3.6.3

    ## Loading required package: NLP

``` r
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review)) # first step of cleaning the dataset for getting it ready for NLP
as.character(corpus[[1]])
```

    ## [1] "Wow... Loved this place."

``` r
corpus = tm_map(corpus, content_transformer(tolower))  # write all the reviews with lower case. by doing that the same word written with both upper case and lower case will be understood as one word.
as.character(corpus[[1]]) #check if it lowercase
```

    ## [1] "wow... loved this place."

``` r
corpus = tm_map(corpus, removeNumbers) 
as.character(corpus[[1]]) #check if the numbers removed
```

    ## [1] "wow... loved this place."

``` r
corpus = tm_map(corpus, removePunctuation) # removes punctuation
as.character(corpus[[1]]) #check if the punctuations removed
```

    ## [1] "wow loved this place"

``` r
library(SnowballC) #this library is necessary for removing unrelevant words
corpus = tm_map(corpus, removeWords, stopwords()) # removes the unrelevant words for NLP
as.character(corpus[[1]]) # see unrelevant words are removed
```

    ## [1] "wow loved  place"

``` r
corpus = tm_map(corpus, stemDocument) # getting the root of each word
as.character(corpus[[1]])
```

    ## [1] "wow love place"

``` r
corpus = tm_map(corpus, stripWhitespace) # removing extra spaces caused by previous steps
as.character(corpus[[841]])
```

    ## [1] "buck head realli expect better food"

The data is clean now. This dataset has 1000 rows. And assume that this
dataset has 1500 different words.That mean is is going to be 1500
columns and 1000 rows. The dataset become huge now.

``` r
# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus) # this function creates the huge function I talk about above.
dtm # as you see here the sparsity is %100. THe reason is the dataset has too many 0.
```

    ## <<DocumentTermMatrix (documents: 1000, terms: 1577)>>
    ## Non-/sparse entries: 5435/1571565
    ## Sparsity           : 100%
    ## Maximal term length: 32
    ## Weighting          : term frequency (tf)

It has 1577 terms.

``` r
dtm = removeSparseTerms(dtm, 0.999) # filter nonfrequent words of the matrix. We want to keep the %99 of the most frequent words.
dtm
```

    ## <<DocumentTermMatrix (documents: 1000, terms: 691)>>
    ## Non-/sparse entries: 4549/686451
    ## Sparsity           : 99%
    ## Maximal term length: 12
    ## Weighting          : term frequency (tf)

After filtering non ferequent words number of terms reduced to 691. Now
the column number is 691. This function removed nonfrequent words. And
made dataset more predictible and significant. We choose 0.999 because
we have 1000 observations which is not so big. So we choose big number
like 0.999.

``` r
# Since our dataset is matrix, we need to convert it to dataframe for classification models
dataset = as.data.frame(as.matrix(dtm))
head(dataset)
```

    ##   absolut acknowledg actual ago almost also although alway amaz ambianc
    ## 1       0          0      0   0      0    0        0     0    0       0
    ## 2       0          0      0   0      0    0        0     0    0       0
    ## 3       0          0      0   0      0    0        0     0    0       0
    ## 4       0          0      0   0      0    0        0     0    0       0
    ## 5       0          0      0   0      0    0        0     0    0       0
    ## 6       0          0      0   0      0    0        0     0    0       0
    ##   ambienc amount anoth anyon anyth anytim anyway apolog appet area arent
    ## 1       0      0     0     0     0      0      0      0     0    0     0
    ## 2       0      0     0     0     0      0      0      0     0    0     0
    ## 3       0      0     0     0     0      0      0      0     0    0     0
    ## 4       0      0     0     0     0      0      0      0     0    0     0
    ## 5       0      0     0     0     0      0      0      0     0    0     0
    ## 6       0      0     0     0     0      0      0      0     0    0     0
    ##   around arriv ask assur ate atmospher attack attent attitud authent
    ## 1      0     0   0     0   0         0      0      0       0       0
    ## 2      0     0   0     0   0         0      0      0       0       0
    ## 3      0     0   0     0   0         0      0      0       0       0
    ## 4      0     0   0     0   0         0      0      0       0       0
    ## 5      0     0   0     0   0         0      0      0       0       0
    ## 6      0     0   0     0   0         0      0      0       0       0
    ##   averag avoid away awesom awkward babi bachi back bacon bad bagel bakeri
    ## 1      0     0    0      0       0    0     0    0     0   0     0      0
    ## 2      0     0    0      0       0    0     0    0     0   0     0      0
    ## 3      0     0    0      0       0    0     0    0     0   0     0      0
    ## 4      0     0    0      0       0    0     0    0     0   0     0      0
    ## 5      0     0    0      0       0    0     0    0     0   0     0      0
    ## 6      0     0    0      0       0    0     0    0     0   0     0      0
    ##   bar bare bartend basic bathroom batter bay bean beat beauti becom beef
    ## 1   0    0       0     0        0      0   0    0    0      0     0    0
    ## 2   0    0       0     0        0      0   0    0    0      0     0    0
    ## 3   0    0       0     0        0      0   0    0    0      0     0    0
    ## 4   0    0       0     0        0      0   0    0    0      0     0    0
    ## 5   0    0       0     0        0      0   0    0    0      0     0    0
    ## 6   0    0       0     0        0      0   0    0    0      0     0    0
    ##   beer behind believ belli best better beyond big bill biscuit bisqu bit
    ## 1    0      0      0     0    0      0      0   0    0       0     0   0
    ## 2    0      0      0     0    0      0      0   0    0       0     0   0
    ## 3    0      0      0     0    0      0      0   0    0       0     0   0
    ## 4    0      0      0     0    0      0      0   0    0       0     0   0
    ## 5    0      0      0     0    0      0      0   0    0       0     0   0
    ## 6    0      0      0     0    0      0      0   0    0       0     0   0
    ##   bite black bland blow boba boot bother bowl box boy boyfriend bread
    ## 1    0     0     0    0    0    0      0    0   0   0         0     0
    ## 2    0     0     0    0    0    0      0    0   0   0         0     0
    ## 3    0     0     0    0    0    0      0    0   0   0         0     0
    ## 4    0     0     0    0    0    0      0    0   0   0         0     0
    ## 5    0     0     0    0    0    0      0    0   0   0         0     0
    ## 6    0     0     0    0    0    0      0    0   0   0         0     0
    ##   break breakfast brick bring brought brunch buck buffet build burger busi
    ## 1     0         0     0     0       0      0    0      0     0      0    0
    ## 2     0         0     0     0       0      0    0      0     0      0    0
    ## 3     0         0     0     0       0      0    0      0     0      0    0
    ## 4     0         0     0     0       0      0    0      0     0      0    0
    ## 5     0         0     0     0       0      0    0      0     0      0    0
    ## 6     0         0     0     0       0      0    0      0     0      0    0
    ##   butter cafe call came can cant car care cashier char charcoal charg
    ## 1      0    0    0    0   0    0   0    0       0    0        0     0
    ## 2      0    0    0    0   0    0   0    0       0    0        0     0
    ## 3      0    0    0    0   0    0   0    0       0    0        0     0
    ## 4      0    0    0    0   0    0   0    0       0    0        0     0
    ## 5      0    0    0    0   0    0   0    0       0    0        0     0
    ## 6      0    0    0    0   0    0   0    0       0    0        0     0
    ##   cheap check chees cheeseburg chef chewi chicken chines chip choos
    ## 1     0     0     0          0    0     0       0      0    0     0
    ## 2     0     0     0          0    0     0       0      0    0     0
    ## 3     0     0     0          0    0     0       0      0    0     0
    ## 4     0     0     0          0    0     0       0      0    0     0
    ## 5     0     0     0          0    0     0       0      0    0     0
    ## 6     0     0     0          0    0     0       0      0    0     0
    ##   classic clean close cocktail coffe cold color combin combo come comfort
    ## 1       0     0     0        0     0    0     0      0     0    0       0
    ## 2       0     0     0        0     0    0     0      0     0    0       0
    ## 3       0     0     0        0     0    0     0      0     0    0       0
    ## 4       0     0     0        0     0    0     0      0     0    0       0
    ## 5       0     0     0        0     0    0     0      0     0    0       0
    ## 6       0     0     0        0     0    0     0      0     0    0       0
    ##   compani complain complaint complet consid contain conveni cook cool
    ## 1       0        0         0       0      0       0       0    0    0
    ## 2       0        0         0       0      0       0       0    0    0
    ## 3       0        0         0       0      0       0       0    0    0
    ## 4       0        0         0       0      0       0       0    0    0
    ## 5       0        0         0       0      0       0       0    0    0
    ## 6       0        0         0       0      0       0       0    0    0
    ##   correct couldnt coupl cours cover cow crazi cream creami crowd crust
    ## 1       0       0     0     0     0   0     0     0      0     0     0
    ## 2       0       0     0     0     0   0     0     0      0     0     1
    ## 3       0       0     0     0     0   0     0     0      0     0     0
    ## 4       0       0     0     0     0   0     0     0      0     0     0
    ## 5       0       0     0     0     0   0     0     0      0     0     0
    ## 6       0       0     0     0     0   0     0     0      0     0     0
    ##   curri custom cut cute damn dark date day deal decent decid decor defin
    ## 1     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 2     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 3     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 4     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 5     0      0   0    0    0    0    0   0    0      0     0     0     0
    ## 6     0      0   0    0    1    0    0   0    0      0     0     0     0
    ##   definit delici delight delish deserv dessert didnt die differ dine
    ## 1       0      0       0      0      0       0     0   0      0    0
    ## 2       0      0       0      0      0       0     0   0      0    0
    ## 3       0      0       0      0      0       0     0   0      0    0
    ## 4       0      0       0      0      0       0     0   0      0    0
    ## 5       0      0       0      0      0       0     0   0      0    0
    ## 6       0      0       0      0      0       0     0   0      0    0
    ##   dinner dirt dirti disappoint disgrac disgust dish disrespect dog done
    ## 1      0    0     0          0       0       0    0          0   0    0
    ## 2      0    0     0          0       0       0    0          0   0    0
    ## 3      0    0     0          0       0       0    0          0   0    0
    ## 4      0    0     0          0       0       0    0          0   0    0
    ## 5      0    0     0          0       0       0    0          0   0    0
    ## 6      0    0     0          0       0       0    0          0   0    0
    ##   dont door doubl doubt downtown dress dri driest drink drive duck eat
    ## 1    0    0     0     0        0     0   0      0     0     0    0   0
    ## 2    0    0     0     0        0     0   0      0     0     0    0   0
    ## 3    0    0     0     0        0     0   0      0     0     0    0   0
    ## 4    0    0     0     0        0     0   0      0     0     0    0   0
    ## 5    0    0     0     0        0     0   0      0     0     0    0   0
    ## 6    0    0     0     0        0     0   0      0     0     0    0   0
    ##   eaten edibl egg eggplant either els elsewher employe empti end enjoy
    ## 1     0     0   0        0      0   0        0       0     0   0     0
    ## 2     0     0   0        0      0   0        0       0     0   0     0
    ## 3     0     0   0        0      0   0        0       0     0   0     0
    ## 4     0     0   0        0      0   0        0       0     0   0     0
    ## 5     0     0   0        0      0   0        0       0     0   0     0
    ## 6     0     0   0        0      0   0        0       0     0   0     0
    ##   enough entre equal especi establish even event ever everi everyon
    ## 1      0     0     0      0         0    0     0    0     0       0
    ## 2      0     0     0      0         0    0     0    0     0       0
    ## 3      0     0     0      0         0    0     0    0     0       0
    ## 4      0     0     0      0         0    0     0    0     0       0
    ## 5      0     0     0      0         0    0     0    0     0       0
    ## 6      0     0     0      0         0    0     0    0     0       0
    ##   everyth excel excus expect experi experienc extra extrem eye fact fail
    ## 1       0     0     0      0      0         0     0      0   0    0    0
    ## 2       0     0     0      0      0         0     0      0   0    0    0
    ## 3       0     0     0      0      0         0     0      0   0    0    0
    ## 4       0     0     0      0      0         0     0      0   0    0    0
    ## 5       0     0     0      0      0         0     0      0   0    0    0
    ## 6       0     0     0      0      0         0     0      0   0    0    0
    ##   fair famili familiar fan fantast far fare fast favor favorit feel fell
    ## 1    0      0        0   0       0   0    0    0     0       0    0    0
    ## 2    0      0        0   0       0   0    0    0     0       0    0    0
    ## 3    0      0        0   0       0   0    0    0     0       0    0    0
    ## 4    0      0        0   0       0   0    0    0     0       0    0    0
    ## 5    0      0        0   0       0   0    0    0     0       0    0    0
    ## 6    0      0        0   0       0   0    0    0     0       0    0    0
    ##   felt filet fill final find fine finish first fish flavor flavorless
    ## 1    0     0    0     0    0    0      0     0    0      0          0
    ## 2    0     0    0     0    0    0      0     0    0      0          0
    ## 3    0     0    0     0    0    0      0     0    0      0          0
    ## 4    0     0    0     0    0    0      0     0    0      0          0
    ## 5    0     0    0     0    0    0      0     0    0      0          0
    ## 6    0     0    0     0    0    0      0     0    0      0          0
    ##   flower focus folk food found fresh fri friend front frozen full fun
    ## 1      0     0    0    0     0     0   0      0     0      0    0   0
    ## 2      0     0    0    0     0     0   0      0     0      0    0   0
    ## 3      0     0    0    0     0     0   0      0     0      0    0   0
    ## 4      0     0    0    0     0     0   0      0     0      0    0   0
    ## 5      0     0    0    0     0     0   0      0     0      0    0   0
    ## 6      0     0    0    0     0     0   0      0     0      0    0   0
    ##   garlic gave generous get give given glad gold gone good got greas great
    ## 1      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 2      0    0        0   0    0     0    0    0    0    1   0     0     0
    ## 3      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 4      0    0        0   0    0     0    0    0    0    0   0     0     0
    ## 5      0    0        0   0    0     0    0    0    0    0   0     0     1
    ## 6      0    0        0   1    0     0    0    0    0    0   0     0     0
    ##   greek green greet grill gross group guess guest guy gyro hair half hand
    ## 1     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 2     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 3     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 4     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 5     0     0     0     0     0     0     0     0   0    0    0    0    0
    ## 6     0     0     0     0     0     0     0     0   0    0    0    0    0
    ##   handl happen happi hard hate head healthi heard heart heat help high
    ## 1     0      0     0    0    0    0       0     0     0    0    0    0
    ## 2     0      0     0    0    0    0       0     0     0    0    0    0
    ## 3     0      0     0    0    0    0       0     0     0    0    0    0
    ## 4     0      0     0    0    0    0       0     0     0    0    0    0
    ## 5     0      0     0    0    0    0       0     0     0    0    0    0
    ## 6     0      0     0    0    0    0       0     0     0    0    0    0
    ##   highlight hit home homemad honest hope horribl hot hour hous howev huge
    ## 1         0   0    0       0      0    0       0   0    0    0     0    0
    ## 2         0   0    0       0      0    0       0   0    0    0     0    0
    ## 3         0   0    0       0      0    0       0   0    0    0     0    0
    ## 4         0   0    0       0      0    0       0   0    0    0     0    0
    ## 5         0   0    0       0      0    0       0   0    0    0     0    0
    ## 6         0   0    0       0      0    0       0   0    0    0     0    0
    ##   human hummus husband ice ignor ill imagin immedi impecc impress includ
    ## 1     0      0       0   0     0   0      0      0      0       0      0
    ## 2     0      0       0   0     0   0      0      0      0       0      0
    ## 3     0      0       0   0     0   0      0      0      0       0      0
    ## 4     0      0       0   0     0   0      0      0      0       0      0
    ## 5     0      0       0   0     0   0      0      0      0       0      0
    ## 6     0      0       0   0     0   0      0      0      0       0      0
    ##   incred indian inexpens insid insult interest isnt italian ive job joint
    ## 1      0      0        0     0      0        0    0       0   0   0     0
    ## 2      0      0        0     0      0        0    0       0   0   0     0
    ## 3      0      0        0     0      0        0    0       0   0   0     0
    ## 4      0      0        0     0      0        0    0       0   0   0     0
    ## 5      0      0        0     0      0        0    0       0   0   0     0
    ## 6      0      0        0     0      0        0    0       0   0   0     0
    ##   joke judg just kept kid kind know known lack ladi larg last late later
    ## 1    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 2    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 3    0    0    1    0   0    0    0     0    0    0    0    0    0     0
    ## 4    0    0    0    0   0    0    0     0    0    0    0    0    1     0
    ## 5    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ## 6    0    0    0    0   0    0    0     0    0    0    0    0    0     0
    ##   least leav left legit let life light like list liter littl live lobster
    ## 1     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 2     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 3     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 4     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 5     0    0    0     0   0    0     0    0    0     0     0    0       0
    ## 6     0    0    0     0   0    0     0    0    0     0     0    0       0
    ##   locat long longer look lost lot love lover lukewarm lunch made main make
    ## 1     0    0      0    0    0   0    1     0        0     0    0    0    0
    ## 2     0    0      0    0    0   0    0     0        0     0    0    0    0
    ## 3     0    0      0    0    0   0    0     0        0     0    0    0    0
    ## 4     0    0      0    0    0   0    1     0        0     0    0    0    0
    ## 5     0    0      0    0    0   0    0     0        0     0    0    0    0
    ## 6     0    0      0    0    0   0    0     0        0     0    0    0    0
    ##   mall manag mani margarita mari may mayb meal mean meat mediocr meh melt
    ## 1    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 2    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 3    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 4    0     0    0         0    0   1    0    0    0    0       0   0    0
    ## 5    0     0    0         0    0   0    0    0    0    0       0   0    0
    ## 6    0     0    0         0    0   0    0    0    0    0       0   0    0
    ##   menu mexican mid min mind minut miss mistak moist mom money mood mouth
    ## 1    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 2    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 3    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 4    0       0   0   0    0     0    0      0     0   0     0    0     0
    ## 5    1       0   0   0    0     0    0      0     0   0     0    0     0
    ## 6    0       0   0   0    0     0    0      0     0   0     0    0     0
    ##   much multipl mushroom music must nacho nasti need needless neighborhood
    ## 1    0       0        0     0    0     0     0    0        0            0
    ## 2    0       0        0     0    0     0     0    0        0            0
    ## 3    0       0        0     0    0     0     1    0        0            0
    ## 4    0       0        0     0    0     0     0    0        0            0
    ## 5    0       0        0     0    0     0     0    0        0            0
    ## 6    0       0        0     0    0     0     0    0        0            0
    ##   never new next nice nicest night none note noth now offer old omg one
    ## 1     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 2     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 3     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 4     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 5     0   0    0    0      0     0    0    0    0   0     0   0   0   0
    ## 6     0   0    0    0      0     0    0    0    0   1     0   0   0   0
    ##   opportun option order other outsid outstand oven overal overcook overpr
    ## 1        0      0     0     0      0        0    0      0        0      0
    ## 2        0      0     0     0      0        0    0      0        0      0
    ## 3        0      0     0     0      0        0    0      0        0      0
    ## 4        0      0     0     0      0        0    0      0        0      0
    ## 5        0      0     0     0      0        0    0      0        0      0
    ## 6        0      0     0     0      0        0    0      0        0      0
    ##   overwhelm owner pace pack paid pancak paper par part parti pass pasta
    ## 1         0     0    0    0    0      0     0   0    0     0    0     0
    ## 2         0     0    0    0    0      0     0   0    0     0    0     0
    ## 3         0     0    0    0    0      0     0   0    0     0    0     0
    ## 4         0     0    0    0    0      0     0   0    0     0    0     0
    ## 5         0     0    0    0    0      0     0   0    0     0    0     0
    ## 6         0     0    0    0    0      0     0   0    0     0    0     0
    ##   patio pay peanut peopl perfect person pho phoenix pictur piec pita pizza
    ## 1     0   0      0     0       0      0   0       0      0    0    0     0
    ## 2     0   0      0     0       0      0   0       0      0    0    0     0
    ## 3     0   0      0     0       0      0   0       0      0    0    0     0
    ## 4     0   0      0     0       0      0   0       0      0    0    0     0
    ## 5     0   0      0     0       0      0   0       0      0    0    0     0
    ## 6     0   0      0     0       0      0   1       0      0    0    0     0
    ##   place plate play pleas pleasant plus point poor pop pork portion possibl
    ## 1     1     0    0     0        0    0     0    0   0    0       0       0
    ## 2     0     0    0     0        0    0     0    0   0    0       0       0
    ## 3     0     0    0     0        0    0     0    0   0    0       0       0
    ## 4     0     0    0     0        0    0     0    0   0    0       0       0
    ## 5     0     0    0     0        0    0     0    0   0    0       0       0
    ## 6     0     0    0     0        0    0     0    0   0    0       0       0
    ##   potato prepar present pretti price probabl profession promis prompt
    ## 1      0      0       0      0     0       0          0      0      0
    ## 2      0      0       0      0     0       0          0      0      0
    ## 3      0      0       0      0     0       0          0      0      0
    ## 4      0      0       0      0     0       0          0      0      0
    ## 5      0      0       0      0     1       0          0      0      0
    ## 6      0      0       0      0     0       0          0      0      0
    ##   provid public pull pure put qualiti quick quit rare rate rather rave
    ## 1      0      0    0    0   0       0     0    0    0    0      0    0
    ## 2      0      0    0    0   0       0     0    0    0    0      0    0
    ## 3      0      0    0    0   0       0     0    0    0    0      0    0
    ## 4      0      0    0    0   0       0     0    0    0    0      0    0
    ## 5      0      0    0    0   0       0     0    0    0    0      0    0
    ## 6      0      0    0    0   0       0     0    0    0    0      0    0
    ##   read real realiz realli reason receiv recent recommend red refil regular
    ## 1    0    0      0      0      0      0      0         0   0     0       0
    ## 2    0    0      0      0      0      0      0         0   0     0       0
    ## 3    0    0      0      0      0      0      0         0   0     0       0
    ## 4    0    0      0      0      0      0      0         1   0     0       0
    ## 5    0    0      0      0      0      0      0         0   0     0       0
    ## 6    0    0      0      0      0      0      0         0   0     0       0
    ##   relax remind restaur return review rice right roast roll room rude run
    ## 1     0      0       0      0      0    0     0     0    0    0    0   0
    ## 2     0      0       0      0      0    0     0     0    0    0    0   0
    ## 3     0      0       0      0      0    0     0     0    0    0    0   0
    ## 4     0      0       0      0      0    0     0     0    0    0    0   0
    ## 5     0      0       0      0      0    0     0     0    0    0    0   0
    ## 6     0      0       0      0      0    0     0     0    0    0    0   0
    ##   sad said salad salmon salsa salt sandwich sashimi sat satisfi sauc say
    ## 1   0    0     0      0     0    0        0       0   0       0    0   0
    ## 2   0    0     0      0     0    0        0       0   0       0    0   0
    ## 3   0    0     0      0     0    0        0       0   0       0    0   0
    ## 4   0    0     0      0     0    0        0       0   0       0    0   0
    ## 5   0    0     0      0     0    0        0       0   0       0    0   0
    ## 6   0    0     0      0     0    0        0       0   0       0    0   0
    ##   scallop seafood season seat second see seem seen select serious serv
    ## 1       0       0      0    0      0   0    0    0      0       0    0
    ## 2       0       0      0    0      0   0    0    0      0       0    0
    ## 3       0       0      0    0      0   0    0    0      0       0    0
    ## 4       0       0      0    0      0   0    0    0      0       0    0
    ## 5       0       0      0    0      0   0    0    0      1       0    0
    ## 6       0       0      0    0      0   0    0    0      0       0    0
    ##   server servic set sever shop show shrimp sick side sign similar simpl
    ## 1      0      0   0     0    0    0      0    0    0    0       0     0
    ## 2      0      0   0     0    0    0      0    0    0    0       0     0
    ## 3      0      0   0     0    0    0      0    0    0    0       0     0
    ## 4      0      0   0     0    0    0      0    0    0    0       0     0
    ## 5      0      0   0     0    0    0      0    0    0    0       0     0
    ## 6      0      0   0     0    0    0      0    0    0    0       0     0
    ##   simpli sinc singl sit six slice slow small smell soggi someon someth
    ## 1      0    0     0   0   0     0    0     0     0     0      0      0
    ## 2      0    0     0   0   0     0    0     0     0     0      0      0
    ## 3      0    0     0   0   0     0    0     0     0     0      0      0
    ## 4      0    0     0   0   0     0    0     0     0     0      0      0
    ## 5      0    0     0   0   0     0    0     0     0     0      0      0
    ## 6      0    0     0   0   0     0    0     0     0     0      0      0
    ##   soon soooo sore soup special spend spice spici spot staff stale star
    ## 1    0     0    0    0       0     0     0     0    0     0     0    0
    ## 2    0     0    0    0       0     0     0     0    0     0     0    0
    ## 3    0     0    0    0       0     0     0     0    0     0     0    0
    ## 4    0     0    0    0       0     0     0     0    0     0     0    0
    ## 5    0     0    0    0       0     0     0     0    0     0     0    0
    ## 6    0     0    0    0       0     0     0     0    0     0     0    0
    ##   start station stay steak step stick still stir stomach stop strip stuf
    ## 1     0       0    0     0    0     0     0    0       0    0     0    0
    ## 2     0       0    0     0    0     0     0    0       0    0     0    0
    ## 3     0       0    0     0    0     0     0    0       0    0     0    0
    ## 4     0       0    0     0    0     0     0    0       0    1     0    0
    ## 5     0       0    0     0    0     0     0    0       0    0     0    0
    ## 6     0       0    0     0    0     0     0    0       0    0     0    0
    ##   stuff style subpar subway suck sugari suggest summer super sure surpris
    ## 1     0     0      0      0    0      0       0      0     0    0       0
    ## 2     0     0      0      0    0      0       0      0     0    0       0
    ## 3     0     0      0      0    0      0       0      0     0    0       0
    ## 4     0     0      0      0    0      0       0      0     0    0       0
    ## 5     0     0      0      0    0      0       0      0     0    0       0
    ## 6     0     0      0      0    0      0       0      0     0    0       0
    ##   sushi sweet tabl taco take talk tap tapa tartar tast tasteless tasti tea
    ## 1     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 2     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 3     0     0    0    0    0    0   0    0      0    0         0     1   0
    ## 4     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 5     0     0    0    0    0    0   0    0      0    0         0     0   0
    ## 6     0     0    0    0    0    0   0    0      0    0         0     0   0
    ##   tell ten tender terribl textur thai that thin thing think third though
    ## 1    0   0      0       0      0    0    0    0     0     0     0      0
    ## 2    0   0      0       0      0    0    0    0     0     0     0      0
    ## 3    0   0      0       0      1    0    0    0     0     0     0      0
    ## 4    0   0      0       0      0    0    0    0     0     0     0      0
    ## 5    0   0      0       0      0    0    0    0     0     0     0      0
    ## 6    0   0      0       0      0    0    0    0     0     0     0      0
    ##   thought thumb time tip toast today told took top tot total touch toward
    ## 1       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 2       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 3       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 4       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 5       0     0    0   0     0     0    0    0   0   0     0     0      0
    ## 6       0     0    0   0     0     0    0    0   0   0     0     0      0
    ##   town treat tri trip tuna twice two unbeliev undercook underwhelm
    ## 1    0     0   0    0    0     0   0        0         0          0
    ## 2    0     0   0    0    0     0   0        0         0          0
    ## 3    0     0   0    0    0     0   0        0         0          0
    ## 4    0     0   0    0    0     0   0        0         0          0
    ## 5    0     0   0    0    0     0   0        0         0          0
    ## 6    0     0   0    0    0     0   0        0         0          0
    ##   unfortun unless use valley valu vega veget vegetarian ventur vibe
    ## 1        0      0   0      0    0    0     0          0      0    0
    ## 2        0      0   0      0    0    0     0          0      0    0
    ## 3        0      0   0      0    0    0     0          0      0    0
    ## 4        0      0   0      0    0    0     0          0      0    0
    ## 5        0      0   0      0    0    0     0          0      0    0
    ## 6        0      0   0      0    0    0     0          0      0    0
    ##   vinegrett visit wait waiter waitress walk wall want warm wasnt wast
    ## 1         0     0    0      0        0    0    0    0    0     0    0
    ## 2         0     0    0      0        0    0    0    0    0     0    0
    ## 3         0     0    0      0        0    0    0    0    0     0    0
    ## 4         0     0    0      0        0    0    0    0    0     0    0
    ## 5         0     0    0      0        0    0    0    0    0     0    0
    ## 6         0     0    0      0        0    0    0    1    0     0    0
    ##   watch water way week well went weve white whole wife will wine wing
    ## 1     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 2     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 3     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 4     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 5     0     0   0    0    0    0    0     0     0    0    0    0    0
    ## 6     0     0   0    0    0    0    0     0     0    0    0    0    0
    ##   without wonder wont word work worker world wors worst worth wouldnt wow
    ## 1       0      0    0    0    0      0     0    0     0     0       0   1
    ## 2       0      0    0    0    0      0     0    0     0     0       0   0
    ## 3       0      0    0    0    0      0     0    0     0     0       0   0
    ## 4       0      0    0    0    0      0     0    0     0     0       0   0
    ## 5       0      0    0    0    0      0     0    0     0     0       0   0
    ## 6       0      0    0    0    0      0     0    0     0     0       0   0
    ##   wrap wrong year yet youd your yummi zero
    ## 1    0     0    0   0    0    0     0    0
    ## 2    0     0    0   0    0    0     0    0
    ## 3    0     0    0   0    0    0     0    0
    ## 4    0     0    0   0    0    0     0    0
    ## 5    0     0    0   0    0    0     0    0
    ## 6    0     0    0   0    0    0     0    0

As you see the table above there are many words are at the columns.

``` r
dataset$Liked = dataset_original$Liked # we created a new column which has the same name with original one. (liked column)
```

Now, it is time to build classification model. The most successful
classification models for NLP are Decision Tree,Random Forest and Naive
Bayes.

I will continuewith Random Forest

—————-Random Forest Classification———————-

``` r
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
```

We also do not need feature scaling. We have already zeros and ones.

Splitting the dataset into the Training set and Test set

``` r
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
```

``` r
# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
classifier = randomForest(x = training_set[-692], # training set shouldn't include  dependent variable
                          y = training_set$Liked,
                          ntree = 10) 
# I'll try with 10 trees. But you can try with more trees and compare the accuracy of them to figure out most accurate result.
```

``` r
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
y_pred
```

    ##   4   9  10  16  17  21  24  33  39  40  41  48  56  58  59  61  63  73 
    ##   1   1   1   0   0   0   0   0   1   0   0   1   1   0   1   0   1   0 
    ##  76  82  92  93  98  99 105 112 113 115 116 122 123 142 150 152 154 157 
    ##   0   0   1   0   0   1   0   0   1   1   0   0   1   1   0   1   0   1 
    ## 158 159 161 169 182 183 184 188 190 191 193 199 202 203 210 211 217 222 
    ##   1   1   0   0   0   0   0   0   1   1   0   1   1   0   0   0   1   0 
    ## 228 239 240 250 251 255 258 262 264 270 272 276 287 292 303 306 314 318 
    ##   1   0   1   0   1   1   0   0   1   0   0   0   0   0   0   0   0   0 
    ## 326 328 337 344 345 346 349 351 353 361 363 364 370 375 395 396 397 399 
    ##   0   1   0   0   0   1   0   0   0   0   0   1   0   1   1   0   0   1 
    ## 412 413 415 416 430 433 445 446 453 456 466 469 470 473 486 495 496 509 
    ##   1   0   0   0   1   1   1   1   1   0   1   1   1   1   1   0   0   0 
    ## 519 521 525 528 531 535 539 545 548 555 560 563 568 570 574 583 586 591 
    ##   0   1   1   1   1   0   0   0   0   1   0   1   1   0   1   0   1   1 
    ## 598 606 613 614 618 625 628 633 634 639 641 647 648 653 658 668 674 679 
    ##   1   1   0   0   0   1   1   0   1   0   1   0   1   1   1   1   1   1 
    ## 688 694 698 712 715 716 719 730 739 743 752 759 761 768 780 789 795 807 
    ##   1   1   0   0   1   1   0   1   1   0   1   1   1   1   1   1   0   0 
    ## 809 811 817 818 821 844 848 849 853 855 863 868 874 882 890 891 892 894 
    ##   1   1   0   1   0   0   1   0   1   1   0   0   1   0   1   1   1   0 
    ## 900 905 906 912 915 920 924 931 935 938 939 941 953 956 965 973 977 983 
    ##   1   0   1   0   0   1   0   1   0   0   1   0   1   0   0   0   0   0 
    ## 985 996 
    ##   0   0 
    ## Levels: 0 1

``` r
# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
accuracy = (82+77) / (82+77+23+18)
accuracy 
```

    ## [1] 0.795

The accuracy of the model is 0.795
