from tensorflow.keras.backend import flatten, sum

class Metrics:

    def __init__(self):
        self.smooth = 1.0

    def dice_coef(self, y_true, y_pred):
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        intersection = sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (sum(y_true_f) + sum(y_pred_f) + self.smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return -self.dice_coef(y_true, y_pred)