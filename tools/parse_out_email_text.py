from nltk.stem.snowball import SnowballStemmer
import string

def parse_out_text(f):
    f.seek(0)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        word_list = text_string.split()
        stemmer = SnowballStemmer("english")
        for word in word_list:
            words += stemmer.stem(word) + " "

    return words

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parse_out_text(ff)
    print text

if __name__ == '__main__':
    main()

