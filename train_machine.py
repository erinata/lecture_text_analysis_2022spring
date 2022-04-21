import nltk

# nltk.download("stopwords")

from nltk.corpus import stopwords
# print(stopwords.words("english"))

import pandas
import json

import kfold_template

import string

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix, accuracy_score

# import pickle
import dill as pickle


review_stars = []
review_text = []

with open('yelp_review_part.json', encoding="utf-8") as f:
	for line in f:
		json_line = json.loads(line)
		review_stars.append(json_line["stars"])
		review_text.append(json_line["text"])

dataset = pandas.DataFrame(data = {"text": review_text, "stars": review_stars})

print(dataset.shape)

dataset = dataset[0:500]

# # # If we group 2 3 4 into one group
# dataset["stars"] = dataset["stars"].replace(2, 3)
# dataset["stars"] = dataset["stars"].replace(4, 3)

# # # If we give up 2 and 4
dataset = dataset[(dataset['stars']==1)|(dataset['stars']==3)|(dataset['stars']==5)]
dataset.reset_index(drop=True, inplace=True)


print(dataset.shape)

data = dataset["text"]
target = dataset["stars"]

lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text_processed = text.translate(str.maketrans('', '', string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stopwords.words("english"):
			word_processed = lemmatizer.lemmatize(word_processed)
			result.append(word_processed)
	return result

 
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_transformer.transform(data)

print(data)
print(data.shape)


machine = MultinomialNB()

results = kfold_template.run_kfold(data, target, 4, machine, 1, 1)
print(results[1])
for i in results[2]:
	print(i)



machine = MultinomialNB()
machine.fit(data, target)


# with open("machine.pickle", "wb") as f:
# 	pickle.dump(machine, f)


# with open("count_vectorize_transformer.pickle", "wb") as f:
# 	pickle.dump(count_vectorize_transformer, f)


# with open("lemmatizer.pickle", "wb") as f:
# 	pickle.dump(lemmatizer, f)

# with open("stopwords.pickle", "wb") as f:
# 	pickle.dump(stopwords, f)


# with open("string.pickle", "wb") as f:
# 	pickle.dump(string, f)

with open("objects.pickle", "wb") as f:
	pickle.dump(machine, f)
	pickle.dump(count_vectorize_transformer, f)
	pickle.dump(lemmatizer, f)
	pickle.dump(stopwords, f)
	pickle.dump(string, f)
	pickle.dump(pre_processing, f)















