from mlp import Mlp

class ClassificationHead(Mlp):
    """ the classification head for predicting the class of each point.
    returns the logits for each class
    """

class LocalizationHead(Mlp):
    """regress the delta values for each of the 7 dims describing the bounding 
    box
    """