import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """class for extra attributes (CombinedAttributesAdder).

    Class adds three columns rooms_per_household,population_per_household,
    bedrooms_per_room and removes if asked.

    Parameters
    ----------
    indices : list
              contains required indices to fetch columns
    add_bedrooms_per_room : Boolean
                            takes True or False as per requirement
    """

    def __init__(self, indices, add_bedrooms_per_room):
        """Self-reference method.

        function adds three columns rooms_per_household,population_per_household,
        bedrooms_per_room

        Parameters
        ----------
        indices : list
                  contains required indices to fetch columns
        add_bedrooms_per_room : Boolean
                                takes True or False as per requirement
        """
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.indices = indices

    def fit(self, X, y=None):
        """Self reference method.

        function returns a reference to the instance object on which it was called.

        Parameters
        ----------
        X : Dataframe
            Housing train dataset
        y : Dataframe
            Target Dataset
        """
        return self  # nothing else to do

    def transform(self, X):
        """Column addition using transform.

        function adds three columns rooms_per_household,population_per_household,
        bedrooms_per_room


        Parameters
        ----------
        X : DataFrame
            Housing train dataset
        Returns
        -------
          : Numpy.ndarray
            returns a concatenated numpy array with added feature columns.
        """
        rooms_ix = self.indices[0]
        bedrooms_ix = self.indices[1]
        population_ix = self.indices[2]
        households_ix = self.indices[3]
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def inverse_transform(self, X):
        """Functiom to remove columns.

        function returns the dataframe removing the three columns being
        added in transform function

        Parameters
        ----------
        X : DataFrame
            Housing train dataset
        Returns
        ----------
        X : Numpy.ndarray
            returns original array after removal of extra attributes
        """
        if self.add_bedrooms_per_room:
            X = np.delete(X, -1, axis=1)
            X = np.delete(X, -1, axis=1)
            X = np.delete(X, -1, axis=1)
            return X
        else:
            X = np.delete(X, -1, axis=1)
            X = np.delete(X, -1, axis=1)
            return X
