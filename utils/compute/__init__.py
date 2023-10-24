from .compute import compute_accuracy,accuracy,compute_confusion_matrix,compute_indexes,count_model_predict_digits
from .compute import SCELoss,RCELoss,is_singular_matrix
__all__ = [
    'compute_accuracy', 'accuracy', 'compute_confusion_matrix', 'compute_indexes','count_model_predict_digits',
    "SCELoss","RCELoss","is_singular_matrix"
    ]