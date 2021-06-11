from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def bayes(data, alpha_values, max_features_values):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target_3)

    parameters = {'vectorizer__max_features': max_features_values, 'classifier__alpha': alpha_values}
    pipeline = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')), ('classifier', MultinomialNB())])

    grid = GridSearchCV(pipeline, parameters, cv=10, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_params = grid.best_estimator_.get_params()
    print(f"Best parameters: (alpha: {best_params['classifier__alpha']}), (max_features: {best_params['vectorizer__max_features']})")

    print(f"Score on test data: {grid.best_estimator_.score(X_test, y_test)}")


def svm(data, classifier__c, max_features_values):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target_3)

    parameters = {'vectorizer__max_features': max_features_values, 'classifier__C': classifier__c}
    pipeline = Pipeline([('vectorizer', TfidfVectorizer(stop_words='english')), ('classifier', SVC())])

    grid = GridSearchCV(pipeline, parameters, cv=10, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_params = grid.best_estimator_.get_params()
    print(
        f"Best parameters: (C: {best_params['classifier__C']}), (max_features: {best_params['vectorizer__max_features']})")

    print(f"Score on test data: {grid.best_estimator_.score(X_test, y_test)}")


def run_algorithms(data, alpha_values, c_values, k_values):
    print("BAYES TEST:")
    for alpha in alpha_values:
        for k in k_values:
            bayes(data, alpha, k)

    print("SVM TEST:")
    for c in c_values:
        for k in k_values:
            svm(data, c, k)



    '''reviews = data_processing(data.data)
    classifier = MultinomialNB(alpha=alpha)
    reviews = extract_features(reviews, data.target_3, k=k)
    scores = cross_val_score(classifier, reviews, data.target_3,
                             cv=ShuffleSplit(n_splits=10, test_size=0.1, random_state=0))
    print(f"K: {k}, Alpha: {alpha} : (Mean score: {np.average(scores)}, Best Score: {max(scores)})")'''