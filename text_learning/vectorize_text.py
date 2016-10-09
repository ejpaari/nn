import os
import pickle
import re
import sys
sys.path.append( "../tools/" )
from parse_out_email_text import parse_out_text

from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")
from_data = []
word_data = []
temp_counter = 0
remove_words = ["sara", "shackleton", "chris", "germani"]

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
#        temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")
            parsed_email = parse_out_text(email)

            for remove_word in remove_words:
                parsed_email = parsed_email.replace(remove_word, "")

            word_data.append(parsed_email)
            from_data.append((0 if name == "sara" else 1))

            email.close()

print "emails processed"
print word_data[152]

from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "w"))
pickle.dump(from_data, open("your_email_authors.pkl", "w"))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(word_data)
vectorizer.transform(word_data)
print "Number of words:",len(vectorizer.get_feature_names())
print vectorizer.get_feature_names()[34597]
