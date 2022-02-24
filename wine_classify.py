"""
Visual reports for classification models using matplotlib and scikit-learn.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import precision_recall_fscore_support


# The names of the classification report fields
SCORES_KEYS = ('precision', 'recall', 'f1', 'support')


class ClassificationReport(object):
    """
    The ClassificationReport is a visual tool that wraps a scikit-learn classifier to
    produce a heatmap when score is called. It implements the scikit-learn API so that
    it can easily fit into a sklearn workflow.
    Parameters
    ----------
    model : estimator
        The scikit-learn classifier to draw the classification report for.
    ax : Axes, default=None
        The matplotlib.Axes object to draw the classification report heatmap on.
    cmap : str or cmap, default=plt.cm.Blues
        The color map to use for the heatmap drawing.
    """

    def __init__(self, model, ax=None, cmap=plt.cm.Blues):
        self.model = model
        self._ax = ax
        self.cmap = cmap

    @property
    def ax(self):
        """
        Returns the axes object or creates one if it doesn't exist.
        """
        if self._ax is None:
            _, self._ax = plt.subplots()
        return self._ax

    def fit(self, X_train, y_train):
        """
        Fits the model if it isn't already fitted.
        """
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self

    def score(self, X_test, y_test):
        """
        Score computes the classification report and draws it.
        """
        y_pred = self.model.predict(X_test)
        scores = precision_recall_fscore_support(y_test, y_pred)

        # Create a mapping of metric to class label
        scores = map(lambda s: dict(zip(self.classes_, s)), scores)
        self.scores_ = dict(zip(SCORES_KEYS, scores))

        self.draw()

    def draw(self):
        """
        Renders the classification report
        """
        # Create display grid
        cr_display = np.zeros((len(self.classes_), len(SCORES_KEYS)))

        # For each class row, append columns for metrics
        for idx, cls in enumerate(self.classes_):
            for jdx, metric in enumerate(SCORES_KEYS):
                cr_display[idx, jdx] = self.scores_[metric][cls]

        # Set up the dimensions of the pcolormesh
        # NOTE: pcolormesh accepts grids that are (N+1,M+1)
        X, Y = np.arange(len(self.classes_)+1), np.arange(len(SCORES_KEYS)+1)
        self.ax.set_ylim(bottom=0, top=cr_display.shape[0])
        self.ax.set_xlim(left=0, right=cr_display.shape[1])

        # Draw the heatmap with colors bounded by the min and max of the grid
        # NOTE: I do not understand why this is Y, X instead of X, Y it works
        # in this order but raises an exception with the other order.
        g = self.ax.pcolormesh(
            Y, X, cr_display, vmin=0, vmax=1, cmap=self.cmap, edgecolor='w',
        )

        # Add the color bar
        plt.colorbar(g, ax=self.ax)

        # Set the title of the classifiation report
        self.ax.set_title('{} Classification Report'.format(self.model.__class__.__name__))

        # Set the tick marks appropriately
        self.ax.set_xticks(np.arange(len(SCORES_KEYS))+0.5)
        self.ax.set_yticks(np.arange(len(self.classes_))+0.5)

        self.ax.set_xticklabels(SCORES_KEYS, rotation=45)
        self.ax.set_yticklabels(self.classes_)


def classification_report(model, X, y):
    """
    This helper function creates a classification report visualization from just a model
    and some related data; doing the figure creation and splitting itself.
    """
    _, ax = plt.subplots(figsize=(9,6))
    X_train, X_test, y_train, y_test = tts(X, y)

    viz = ClassificationReport(model, ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)

    plt.show()


if __name__ == "__main__":

    # Import test helpers
    from sklearn.datasets import load_wine
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression

    data = load_wine()
    classification_report(MultinomialNB(), data.data, data.target)
    plt.savefig("MNNB.png")
    classification_report(LogisticRegression(), data.data, data.target)
    plt.savefig("LogReg.png")
