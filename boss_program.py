import pandas
import dill as pickle


with open("objects.pickle", "rb") as f:
	machine = pickle.load(f)
	count_vectorize_transformer = pickle.load(f)
	lemmatizer = pickle.load(f)
	stopwords = pickle.load(f)
	string = pickle.load(f)
	pre_processing = pickle.load(f)
	
	


# # # To make predictions for multiple reviews
new_reviews = pandas.read_csv("new_reviews.csv", header=None)
new_reviews_transformed = count_vectorize_transformer.transform(new_reviews.iloc[:,0])
prediction = machine.predict(new_reviews_transformed)
prediction_prob = machine.predict_proba(new_reviews_transformed)

new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)
new_reviews = pandas.concat([new_reviews, prediction_prob_dataframe], axis=1)

new_reviews = new_reviews.rename(columns={new_reviews.columns[0]:"text", new_reviews.columns[1]:"prediction", new_reviews.columns[2]: "prediction_prob_1", new_reviews.columns[3]: "prediction_prob_3", new_reviews.columns[4]: "prediction_prob_5"})

new_reviews['prediction'] = new_reviews['prediction'].astype(int) 
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'], 5)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'], 5)
new_reviews['prediction_prob_5'] = round(new_reviews['prediction_prob_5'], 5) 


print(new_reviews['prediction'])

# new_reviews.to_csv("new_reviews_with_prediction.csv", index=False, float_format='%.9f')




