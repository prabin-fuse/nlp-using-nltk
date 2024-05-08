# Text Preprocessing:
- Different interpretation and standards could lead to inconsistency in digital languages.
- Machines are sensitive to all those differences and interpret them differently.
- So, proper text preprocessing is needed to standarize all the available texts in dataset.
- Garbage-In-Garbage-Out => So, it is necessary to avoid plug-in good quality data to produce good results.


# Broadly divided into:
## - Cleaning
- lower casing
- html/tags cleaning
- Unicode Normalization (eg. emoji)
- Spelling Check
- Others
## - Basic Preprocessing
### Basic
##### Tokenization
## - Advanced Preprocessing

# Using NLTK
- NLTK stands for Natural Language Toolkit
- Language processing tasks and corresponding NLTK modules with examples of functionality
![image.png](attachment:image.png)


## 1) Lower Case Conversion:
- Processing of converting all the word from corpus to same case. (here, using lower case)
- Don't want our model to get confused by seeing the same word with various casing.
- example => Converting "Processing" to "processing"
- Problems without lower casing:
    - Both words are treated as different one.
    - Increase the vocabulary in word corpus.
    - Higher vector dimension hence requiring more computation.

## 2) Removing unnecessary informations:
- Raw text contains many useless informations like html tags, emoji, punctuations, urls, and other unicode characters

## 3) Tokenization
- Breaking the text into smaller units called tokens.
- Why we need tokens?
    - Each tokens have semantic meaning and can exibit relations among other tokens (relatively coherent)
- a) Word Tokenization:
    - splitting the text on the basis of words.
    - words, numbers, punctuations, and others can be considered as a token
    - use ``` word_tokenize() ``` from ```nltk.tokenize```

- b) Sentence Tokenization:
    - splitting text into sentences
    - Each sentence is treated as token.
    - Usually seperate by full stop (.)
    - main focus is to study the structure of sentence in the analysis
    - user ```sent_tokenize()``` from ```nltk.tokenize```

- c) Regular Expression Tokenization: 
    - Using patterns to split text based on specific rules or conditions.

- See this for more : https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/

## 4) Stop Word Removal:
- Stop words are those words that helps in formation but not in meaning.
- e.g: commonly used words like (a, the, are, and , my)
- These words doesn't posses significant meaning in the document.
- **In POS tagging, stop words are not removed.**
- NLTK library in python has list of stop words stored in 16 different languages.

## 5) Stemming:
- In grammar, **inflection** means the modification of word to express different grammatical categories such as tense, case, voice, aspect, person, number, gender and mood.
- **Stemming** is the process of reducing inflection back to root form.
- Mostly used in infomation Retrevial System
- It is algorithm based (Stemmer algorithm). So, *root word might not belong to the language*
- Examples:
    - root word "like" include:
        - likes
        - liked
        - likely
        - liking
    - root word "fina" include:
        - finally
        - final
        - finalized

- Error in stemming:

- Over Stemming:
    - It happens when two or more unrelated words are reduced to same stem even though they are have different meanings and are not same.
    - Refers to *false positive*.
    - Example:
        - University and universe
        - Some stemming algo may reduce both words to "univers" as root word which is completely wrong.

- Under Stemming: 
    - It happens when two or more related words should be reduced to same root word but are reduced to different stem.
    - Refers to *false negative*
    - Example:
        - data and datum.
        - Some stemming algo may reduce word "data" to "dat" and "datum" to "datu" respectively. This is wrong as both words should have same stem or root word.
        
- Stemming Algorithm in NLTK:
    - Porter Stemmer (for English)
    - Snow Ball Stemmer (Other Language) also called porter 2

## 6) Lemmatization:
- Converts inflected words to their word stem that belongs to original language unlike stemming.
- It is slower that stemming bcz it search in the existing lexical dictionary
- root word => lemma
- search based unlike stemming which was algorithm based.
- Most common lemmatizer is : Wordnet

So, for a same word, the output lemma is different depending upon the POS tagging.

**Problem**:
- Not possible to provide POS tagging to large dataset manually

**Solution**:
- Find out the corresponding POS tags and pass it to the lemmatizer. But lemmatizer only accepts single character which needs to be mapped to appropriate POS tags.

##### In NLTK, POS tagging is done using ```nltk.pos_tag()```
- nltk.pos_tag() only accepts the list even though there is single word.
- returns list of tupple
- 1st element of tupple = word itself 
- 2nd element of tupple = POS tags 


### Difference between stemming and lemmatization:

- Similarity:
    - Both Stemming and Lemmatization reduces the word to its root word form.
- Working Mechanism:
    - Stemming reduces to the base word on the basis of ceratin algorithm 
    - lemmatization reduces on the basis of searching to existing linguistic rules.
- Speed:
    - Stemming is relatively fast as it is based on specific time complexity algorithm
    - Lemmatization is slow as it needs to search for the same word and corresponding lemma
- Availability:
    - Stemming algorithm can be found in many languages.
    - Because of its difficulty in maintaining large dictionary, lemmatization algorithm is available in limited languages.
- Base word:
    - In stemming, base word may not be meaning full as it might not be available in language dictionary
    - In lemmatization, base word is always from the lingistic dictionary. It always makes sense.
    
- Stemmer is easy to build than a lemmatizer as the latter requires deep linguistics knowledge in constructing dictionaries to look up the lemma of the word.

## 7) POS Tagging:
- Part of Speech Tagging is there to avoid confusion between two or more same word at different context of the language.
- Two steps are:
    - Tokenize text (word_tokenize)
    - Apply POS tagging from nltk library ```nltk.pos_tage()```


## 8) Chunking:
Chunking allows us to identify phrases while tokenization allows us to identify words.

Note: 
- A phrase is a word or group of words that works as a single unit to perform a grammatical function.
- Noun phrases are built around a noun.

    Examples:
    - “A planet”
    - “A tilting planet”
    - “A swiftly tilting planet”

- Chunks donot overlap. So one word can only fall under a single chunk.

Before making chunks,
- Part of Speech (POS) tagging should be done. And for POS tagging, tokenization should be done.

For chunking, "Chunk Grammar" must be defined.
- A chunk grammar is a combination of rules on how sentences should be chunked. It often uses regular expressions, or regexes.

Creating chunk parser


For example: This grammar ```NP: {<DT>?<JJ>*<NN>}``` defines :
1. Start with an optional (?) determiner ('DT')
2. Can have any number (*) of adjectives (JJ)
3. End with a noun (<NN>)

## 9) Chinking:
- Chinking is to exclude the pattern while chunking is to include the pattern.

Defining grammar:

```
grammar = """
Chunk: {<.*>+}
       }<JJ>{"""
```

The first rule of your grammar is {<.*>+}. This rule has curly braces that face inward ({}) because it’s used to determine what patterns you want to include in you chunks. In this case, you want to include everything: <.*>+.

The second rule of your grammar is }<JJ>{. This rule has curly braces that face outward (}{) because it’s used to determine what patterns you want to exclude in your chunks. In this case, you want to exclude adjectives: <JJ>.


## 10) Name Entity Recognition (NER)

Named entities are noun phrases that refer to specific locations, people, organiztions etc.

- NER can
    - find named entites in text
    - and determine what type of named entity are they

List of named entity types from NLTK Book:

NE type      | Examples
------------ | ----------------------------------------
ORGANIZATION | Georgia-Pacific Corp., WHO
PERSON       | Eddy Bonte, President Obama
LOCATION     | Murray River, Mount Everest
DATE         | June, 2008-06-29
TIME         | two fifty a m, 1:30 p.m.
MONEY        | 175 million Canadian dollars, GBP 10.40
PERCENT      | twenty pct, 18.75 %
FACILITY     | Washington Monument, Stonehenge
GPE          | South East Asia, Midlothian



Use ```nltk.ne_chunk()``` to determine the named entity.

binary = False:
- gives Named entitry types as well

binary = True:
- only shows if the give tag is Named Entity or not.