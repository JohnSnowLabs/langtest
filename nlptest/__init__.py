try:
    from .robustness.robustness_testing import test_robustness
    from .robustness.robustness_fixing import augment_robustness, test_and_augment_robustness
except:
    print(f'Please run <pip install wn> to use the test_robustness and augment_robustness modules.'
          f'These module are not usable until you install the missing python module.')

from .bias.bias_testing import test_gender_bias
from .noisy_labels.noisy_label_testing import test_label_errors
from .noisy_labels.noisy_label_fixing import InteractiveFix, update_with_model_predictions, add_flag_to_conll
