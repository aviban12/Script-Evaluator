# Script-Evaluator
Problem statement : Develop a machine learning algorithm to predict the score of a script

<pre>
Approach:
	Data Preprocessing
	<br>
		In the provided dataset we mainly need to preprocess EssayText column data because rest of the the data is already in numeric form except clarity and coherant column.
		<br>
		Preprocess EssayText column: 
		<br>
			1 - Replace all the words like i'm,he's with their proper syntax.
			<br>
			2 - Remove all the stop words from the above preprocessed text.
			<br>
			3 - Lemmatize each word using nltk library and also stem each word of corpus.
			<br>
		Prprocess clarity and coherant column:
		<br>	
			1 - Replace "excellent"-3,"above_Average"-2,"average"-1,"worst"-0
</pre>
	Feature Extraction :
		1 - Count number of words ,sentences and average word length of each script.
		2 - Count number of noun, pronoun, adverb, adjective, verb, determiners using textblob library.
		3 - Calculate Flesch Score of each script (not preprocessed).
			Flesch Score -: Flesch score is also known as Flesch reading ease score which predict the ease of readability of given text.
		4 - Calculate Readability consensus based upon seven tests with textstat for each script.
		5 - TFIDF score of each sentence with respect to a particular Essayset.
			TFIDF - Term Frequency inverse document frequency.

	Algorithm Applied:
		1 - Linear Regression gives an accuracy of 68.53 on the provided test cases.
		2 - RandomForesetRegressor gives an accuracy of 72.58 on the provided test cases.
		3 - GradientBoostingRegressor gives an accuracy of 73.08 on the provided test cases.

<pre>
Feature Engineering:
	   Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work.
	   In the given dataset their are two type of data 
		1 - Text Data
		2 - Numeric Data
	  Number of words , Number of sentences ,average word length effect the accuracy of score.
	  Number of Noun, pronoun, adverb, adjective, and deterimers can effect the score od a script.
	  From text data we find out similarity score of sentence with some best clarity and coherent script in the corous. 
	</pre>
<pre>	
IDE Used :
	Pycharm Edu 20

Operating System :
	Linux (ubuntu 18.04)

Tools Used :
	Numpy
	Pandas
	TextBlob
	Gensim
	NLTK
	Sklearn
	re

Exceution : Exeute Run.py file using python3 Run.py command on Linux.
</pre>



