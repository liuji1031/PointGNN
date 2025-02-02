from mlp import Mlp

class ObjectClassificationHead(Mlp):
    """ the classification head for predicting the object class of each point.
    returns the logits for each class
    """

class BackgroundClassificationHead(Mlp):
    """ the binary classification head for predicting the background class of each point.
    returns the logits for each class
    """

class LocalizationHead(Mlp):
    """regress the delta values for each of the 7 dims describing the bounding 
    box
    """