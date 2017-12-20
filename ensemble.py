import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
a = DecisionTreeClassifier()


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.strong_learner = []
        self.alpha = []


    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).
        params:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        return:
            all learning
        '''
        X, y = np.array(X), np.array(y)   # change to array
        print("x shape is:", X.shape, "y shape is:", y.shape)
        if X.shape[0] != y.shape[0]:
            raise Exception("wrong input, x length don't match y")

        # weight of each data
        weight = np.array([float(1) / X.shape[0]] * X.shape[0])
        # all weaker learner
        strong_learner = []
        # weak learner weight
        alpha = []
        # iteration
        for i in range(self.n_weakers_limit):
            weak_learner = self.weak_classifier
            weak_learner.fit(X, y, sample_weight=weight)
            y_pred = weak_learner.predict(X)

            # wrong predict sample

            epsilon = weight[y - y_pred != 0].sum()
            if epsilon > 0.5:
                break
            alp = 0.5 * np.log((1-epsilon)/epsilon)
            temp = weight * np.exp(-1 * alp * y * y_pred)
            weight = temp/temp.sum()

            # append data
            strong_learner.append(weak_learner)
            alpha.append(alp)

            # print predict value
            print("iteration times:", i, classification_report(y, y_pred))

        # final learner
        self.strong_learner = strong_learner
        self.alpha = np.array(alpha)


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''


        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        # strong learner
        y_pred = np.array([learner.predict_proba(X) for learner in self.strong_learner])   # (n_samples, n_weakers_limit)
        y_pred = self.alpha * y_pred
        print(y_pred)

        pass

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
