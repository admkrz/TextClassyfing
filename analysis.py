from statistics import mean
from typing import Dict

from sklearn.feature_extraction.text import CountVectorizer


def analyze_review_number(data):
    class_0 = len([r for r in data.target_3 if r == 0])
    class_1 = len([r for r in data.target_3 if r == 1])
    class_2 = len([r for r in data.target_3 if r == 2])
    class_4_0 = len([r for r in data.target_4 if r == 0])
    class_4_1 = len([r for r in data.target_4 if r == 1])
    class_4_2 = len([r for r in data.target_4 if r == 2])
    class_4_3 = len([r for r in data.target_4 if r == 3])
    return [class_0, class_1, class_2], [class_4_0, class_4_1, class_4_2, class_4_3]


def analyze_review_number_ratings(data):
    length = len(data.data)
    reviews = dict()
    for i in range(0, length):
        if not data.target_ratings[i] in reviews:
            reviews[data.target_ratings[i]] = 1
        else:
            reviews[data.target_ratings[i]] += 1
    return reviews


def analyze_review_length(data):
    length = len(data.target_3)
    lengths_3 = {0: [], 1: [], 2: []}
    lengths_4 = {0: [], 1: [], 2: [], 3: []}
    for i in range(0, length):
        if data.target_3[i] == 0:
            lengths_3[0].append(len(data.data[i]))
        elif data.target_3[i] == 1:
            lengths_3[1].append(len(data.data[i]))
        elif data.target_3[i] == 2:
            lengths_3[2].append(len(data.data[i]))

    length = len(data.target_4)
    for i in range(0, length):
        if data.target_4[i] == 0:
            lengths_4[0].append(len(data.data[i]))
        elif data.target_4[i] == 1:
            lengths_4[1].append(len(data.data[i]))
        elif data.target_4[i] == 2:
            lengths_4[2].append(len(data.data[i]))
        elif data.target_4[i] == 3:
            lengths_4[3].append(len(data.data[i]))

    return [round(mean(lengths_3[0])), round(mean(lengths_3[1])), round(mean(lengths_3[2]))], [
        round(mean(lengths_4[0])), round(mean(lengths_4[1])), round(mean(lengths_4[2])), round(mean(lengths_4[3]))]


def analyze_most_common_words(data):
    length = len(data.target_3)
    reviews = dict()
    print("3 CLASSES:")
    for i in range(0, length):
        if not data.target_3[i] in reviews:
            reviews[data.target_3[i]] = [data.data[i]]
        else:
            reviews[data.target_3[i]].append(data.data[i])
    for i in range(0, 3):
        print(f"CLASS {i}:")
        analyze_most_common_words_for_class([r for rev in reviews[i] for r in rev])

    print("4 CLASSES:")
    for i in range(0, length):
        if not data.target_4[i] in reviews:
            reviews[data.target_4[i]] = [data.data[i]]
        else:
            reviews[data.target_4[i]].append(data.data[i])
    for i in range(0, 4):
        print(f"CLASS {i}:")
        analyze_most_common_words_for_class([r for rev in reviews[i] for r in rev])


def analyze_most_common_words_for_class(list):
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit_transform(list)
    features = vectorizer.get_feature_names()
    words_dict = dict()
    for word in features:
        words_dict[word] = 1
    for word in list:
        if word in words_dict:
            words_dict[word] += 1
    words_dict = dict(sorted(words_dict.items(), key=lambda item: item[1], reverse=True)[:20])
    print(words_dict)
