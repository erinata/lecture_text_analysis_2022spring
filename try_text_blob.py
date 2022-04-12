from textblob import TextBlob

from textblob.sentiments import NaiveBayesAnalyzer


text_blob_object = TextBlob("Excellent! Easy to use. Fast Delivery." , analyzer=NaiveBayesAnalyzer())
print(text_blob_object.sentiment)

text_blob_object = TextBlob("Thank you so much! Thanks for developing this product!" , analyzer=NaiveBayesAnalyzer())
print(text_blob_object.sentiment)




