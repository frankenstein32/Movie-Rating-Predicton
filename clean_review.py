from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

#  init Objects
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# clean Review
def getStemmedReviews(reviews):

	reviews = reviews.lower()
	reviews = reviews.replace('<br /><br />',' ')	

	tokens = tokenizer.tokenize(reviews)
	new_tokens = [token for token in tokens if token not in en_stopwords]
	stemmed_tokens = [ps.stem(token) for token in new_tokens]

	cleaned_review = ' '.join(stemmed_tokens)

	return cleaned_review

def parseLine(review):
	return getStemmedReviews(review)
	