is_sample_core = False

if is_sample_core:
    from TinyPytorch.core_sample import setup_variables
    from TinyPytorch.core_sample import Variable
    from TinyPytorch.core_sample import Function
    from TinyPytorch.core_sample import using_config
    from TinyPytorch.core_sample import no_grad
    from TinyPytorch.core_sample import as_array
    from TinyPytorch.core_sample import as_variable
else:
    from TinyPytorch.core import setup_variables
    from TinyPytorch.core import Variable
    from TinyPytorch.core import Function
    from TinyPytorch.core import using_config
    from TinyPytorch.core import test_mode
    from TinyPytorch.core import no_grad
    from TinyPytorch.core import as_array
    from TinyPytorch.core import as_variable
    from TinyPytorch.core import Config

    import TinyPytorch.datasets
    import TinyPytorch.dataloaders
    import TinyPytorch.optimizers
    import TinyPytorch.functions
    import TinyPytorch.layers
    import TinyPytorch.utils

setup_variables()