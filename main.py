from analysis import analyze_review_length, analyze_review_number, analyze_review_number_ratings, \
    analyze_most_common_words
from data import load_data
from ml_algorithms import bayes, svm, run_algorithms

if __name__ == '__main__':
    data = load_data()
    '''
    print("Number of reviews by class: (3 and 4 classes)")
    print(analyze_review_number(data))
    print("Number of reviews by rating:")
    print(analyze_review_number_ratings(data))
    print("Length of reviews by class: (3 and 4 classes)")
    print(analyze_review_length(data))
    analyze_most_common_words(data)
    '''
    #run_algorithms(data, [0.5, 1, 1.5, 2, 2.5, 3], [0.25, 0.5, 0.75, 1, 2, 5], [20, 50, 75, 100, 200, 500, 1000])
    bayes(data, (0.5, 1, 1.5), (1000, 2500, 4000))
    svm(data, (0.25, 0.5, 0.75), (1000,))

