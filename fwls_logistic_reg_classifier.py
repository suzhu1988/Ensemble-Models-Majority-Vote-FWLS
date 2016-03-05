"""
Construct the FWLS ensemble model using logistic regression
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

class FWLS_Classifier():
    """An FWLS ensemble classifier using logistic regression

    Parameters
    -----------
    classifiers : array-like, shape = [n_classifiers]
      Different classifiers for the ensemble
    
    """
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}

    def fit(self, x, y):
        """Fit classifiers.

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training sampls.

        y : array-like, shape = [n_samples]
            Vector of target class labels.

        Returns
        -------
        self : object

        """
        # Use LabelEncoder to ensure class labels start with 0, which
        # is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(x, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict(self, x):
        """ Predict class labels for x.

        Parameters
        ----------
        x : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.

        Returns
        ----------
        predictions : array-like, shape = [n_samples]
            Predicted class labels.
            
        """
        #Collect results from clf.predict calls
        predictions = np.asarray([clf.predict(x) for clf in self.classifiers_]).T
        return predictions

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(FWLS_Classifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
