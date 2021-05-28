from analysis import analyze_review_length, analyze_review_number, analyze_review_number_ratings, \
    analyze_most_common_words
from data import load_data

if __name__ == '__main__':
    data = load_data()
    print("Number of reviews by class: (3 and 4 classes)")
    print(analyze_review_number(data))
    print("Number of reviews by rating:")
    print(analyze_review_number_ratings(data))
    print("Length of reviews by class: (3 and 4 classes)")
    print(analyze_review_length(data))
    analyze_most_common_words(data)
