import re
import emoji
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

# pip install nltk
# pip install emoji

class BasicCleaning:

    def lower_casing(self, text):
        '''
        Convert the given text into lower case and return the converted one.

        Arguments: 
        text (string) : The raw text that needs to be converted to lower case

        Returns:
        lower_text (string): The lower case converted form oflower text
        '''

        lower_text = text.lower()
        return lower_text
    

    def remove_html(self, text):
        '''
        Detect and remove all the html tags from the string

        Arguments:
        text(string): The raw text from which the html tags needs to be removed

        Returns:
        text_after_html(string) : The text from which html tags are removed.
        '''

        html_pattern = re.compile('<.*?>')
        text_after_html = html_pattern.sub(r'', text)
        return text_after_html
    

    def remote_url(self, text):
        '''
        Detect the urls in text, remove them and return the final text

        Arguments:
        text(string): The raw text form

        Returns:
        text_after_url(string): text which doesn't contain any sorts fo url
        '''

        url_pattern = re.compile(r'https?://\S+|www\. \S+')
        text_after_url = url_pattern.sub(r'', text)
        return text_after_url
    

    def remove_emoji(self, text, replace_with_meaning = True):
        '''
        Detect any sorts of emoji and remove (or replace with meanings) from the original text

        Arguments:
        text(string): raw text that have emojis
        replace_with_meaning(bool) : whether emojis are removed or replaced

        Returns:
        text_after_emoji(string): text after removing the emojis fromt the original text
        '''

        text_after_emoji = emoji.demojize(text)
        return text_after_emoji
    
    def remove_punctuation(self, text):
        '''
        Detects punctuation marks and remove them from the original text

        Arguments:
        text(string): raw text that have punctuations

        Returns:
        text_after_punc(string): text after punctuations are removed.
        '''

        exclude_punc = string.punctuation
        for char in exclude_punc:
            text = text.replace(char, '')
        
        text_after_punc = text
        return text_after_punc
    


class BasicPreprocessing:

    def __init__(self):
        self.basic_clean = BasicCleaning()
        nltk.download('averaged_perceptron_tagger')
        pass


    def tokenize_words(self, text):
        '''
        Accepts string and convert into smaller tokens on the basis of words.

        Arguments:
        text(string) : raw text most probably after basic cleaning is done

        Returns:
        word_tokens(list) : list of word tokens
        '''
        
        word_tokens = word_tokenize(text)
        return word_tokens
    

    def tokenize_sentence(self, text):
        '''
        Accepts string and convert into smaller tokens on the basis of sentence.

        Arguments:
        text(string) : raw text most probably after basic cleaning is done

        Returns:
        sent_tokens(list) : list of word tokens
        '''

        sent_tokens = sent_tokenize(text)
        return sent_tokens

    
    def stop_word_removal(self, text):
        '''
        Removes the stop words from the text and return the filtered text

        Arguments:
        text(string): raw text

        Returns:
        filtered_tokens(list): list of tokens from which stop words have been removed.
        '''
        
        #Tokenize:
        word_tokens = self.tokenize_words(text)

        english_stop_words = stopwords.words('english')
        filtered_tokens = [word for word in word_tokens if word not in english_stop_words]

        return filtered_tokens


    def stemming(self, text):
        '''
        Tokenize the given string on the basis of word tokenization 
        and reduce each tokens back to root word algorithmly

        Arguments:
        text(string) : raw text for stemming purpose

        Returns:
        stem_tokens(list) : list of stemmed tokens
        '''
        #Tokenize:
        word_tokens = self.tokenize_words(text)
        
        # Instantiated PorterStemmer Algorithm
        ps = PorterStemmer()
        stem_tokens = [ps.stem(word) for word in word_tokens]
        return stem_tokens
    

    def get_wordnet_pos(self, word):
        '''
        Maps the POS tag to wordnet POS on the basis of pos_tag() function
        
        Arguments:
        word (string): raw word for which corresponding wordnet POS tag should be find out

        Returns:
        tag_wordnet (char) : character representing noun, verb, adjective, or adverb
        '''

        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
        
        tag_wordnet = tag_dict.get(tag, wordnet.NOUN)

        return tag_wordnet



    def lemmatization(self, text):
        '''
        Tokenize the given string on the basis of word tokenization 
        and reduce each tokens back to root word by searching language dictionary.

        Arguments:
        text(string) : raw text for lemmatization purpose

        Returns:
        lemma_tokens(list) : list of lemmatized tokens
        '''
        #Tokenize:
        word_tokens = self.tokenize_words(text)

        wordnet_lemmatizer = WordNetLemmatizer()
        lemma_tokens = [wordnet_lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in word_tokens]
        
        return lemma_tokens



class AdvancedPreprocessing:

    def __init__(self):
        self.basic_preprocess = BasicPreprocessing()
        nltk.download('averaged_perceptron_tagger')
        # For finding named entity downlaod:
        nltk.download('maxent_ne_chunker')
        nltk.download('words')


    def pos_tagging(self,text):
        '''
        Find out the corresponding Part of Speech Tag for each token in the text

        Arguments:
        text (string): raw text

        Returns:
        tags (list) : list of tupples where 1st element is token and 2nd element is appropriate POS tag.
        '''

        word_tokens = self.basic_preprocess.tokenize_words(text)

        tags = nltk.pos_tag(word_tokens)
        return tags
    

    def named_entity_recognizer(self, text, binary = False):
        '''
        Find out the words that are named entity like people, location, org etc.

        Arguments:
        text(string) : raw text
        binary (bool): Flags whether to give entity a label
                        True => doesn't give label to named entity
                        False => gives the label to named entity (default)
        
        Returns:
        named_entities(list): list of tuples containing all the named entities and corresponsing label either as binary or types of NE                   
        '''

        # Word tokenization
        word_tokens = self.basic_preprocess.tokenize_words(text)
        tags = nltk.pos_tag(word_tokens)

        chunks = nltk.ne_chunk(tags, binary= binary)
        

        # Converting the chunks to list of tupples
        entities =[]
        labels =[]
        for chunk in chunks:
            if hasattr(chunk,'label'): # it returns True if has label NE
                
                entities.append(' '.join(c[0] for c in chunk))
                labels.append(chunk.label())
                
        named_entities = list(set(zip(entities, labels)))

        return named_entities