from text_preprocessing_module import BasicCleaning, BasicPreprocessing, AdvancedPreprocessing

text = '''First and foremost, detecting small objects is hard because small objects are, well, small. 
<h1>The smaller the object</h1>, the less information the detection model has to work with. If a car is far 
off in the distance, it might only occupy a few pixels in our image. In much the same way humans have 
trouble making out distant objects, our model has a harder time identifying cars without visually 
discernible features like wheels and license plates!'''
print(f"Original Text : {text}\n")

# Objects instance:
basic_clean = BasicCleaning()
basic_preprocess = BasicPreprocessing()
advanced_proprocess = AdvancedPreprocessing()

# Lower casing:
lower_text = basic_clean.lower_casing(text)
print(f"The lower cased is : {lower_text}\n")

# Removed html:
text_after_html = basic_clean.remove_html(text)
print(f"The HTML removed text is : {text_after_html}\n")

# Word tokenize
word_tokens = basic_preprocess.tokenize_words(text)
print(f"The word tokenized text is : {word_tokens}\n")

# Sentence tokenize
sent_tokens = basic_preprocess.tokenize_sentence(text)
print(f"The sentence tokenized text is : {sent_tokens}\n")

# Stop word removal
after_stop_words = basic_preprocess.stop_word_removal(text)
print(f"The text after stop word removal is: {after_stop_words}\n")

# Stemming
stem_tokens = basic_preprocess.stemming(text)
print(f"The text after stemming is  : {' '.join(stem_tokens)}\n")

# Lemmatization:
lemma_tokens = basic_preprocess.lemmatization(text)
print(f"The text after lemmatization is : {' '.join(word_tokens)}\n")

# POS tagging:
tags =  advanced_proprocess.pos_tagging(text)
print(f"The corresponding pos tags are : {tags}\n")


text = """
Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879. When he was 17, he moved to Switzerland, where he began his theoretical physics studies at the Swiss Federal Institute of Technology in Zurich. He published his first paper in 1900, at the age of 21.
"""



# Named Entity: with labels
named_entities =  advanced_proprocess.named_entity_recognizer(text)
print(f"The named entity with labels are : {named_entities}\n")

# Named Entity without labels
named_entities =  advanced_proprocess.named_entity_recognizer(text, True)
print(f"The named entity without labels are : {named_entities}")

