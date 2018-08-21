import torch.nn.functional as F


def loss(lambda_reg, lambda_recog):
    return detection_loss(lambda_reg) + lambda_recog * recognition_loss()


def detection_loss(lambda_reg):
    return cls_of_detection_loss() + lambda_reg * reg_of_detection_loss()

def my_loss(y_input, y_target):
    return F.nll_loss(y_input, y_target)

def recognition_loss():
    pass


def cls_of_detection_loss(y_pred, y_label, pixels_mask):
    return F.cross_entropy(y_pred * pixels_mask, y_label * pixels_mask)


def reg_of_detection_loss():
    pass