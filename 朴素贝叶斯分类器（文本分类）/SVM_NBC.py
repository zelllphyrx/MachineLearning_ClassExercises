from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Fetch the 20newsgroups dataset
news = fetch_20newsgroups(subset='all')

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=666)

# Feature extraction: TF-IDF Vectorization
tfidfvec = TfidfVectorizer(stop_words='english', decode_error='ignore')
X_train = tfidfvec.fit_transform(X_train)  # Vectorize training data (text)
X_test = tfidfvec.transform(X_test)  # Vectorize test data (text)

print('Number of features generated =', X_test.shape[1])

# Support Vector Machine classifier
svm = SVC(kernel='linear', C=1.0)  # Instantiate SVM classifier with linear kernel
svm.fit(X_train, y_train)  # Train the classifier
y_pred = svm.predict(X_test)  # Test the classifier

# Print classification report
print(classification_report(y_test, y_pred, target_names=news.target_names))
