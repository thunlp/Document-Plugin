from .output_tool import null_output_function, binary_output_function
from .output_tool import squad_output_function, mlm_output_function

output_function_dic = {
    "Null": null_output_function,
    "binary": binary_output_function,
    "squad": squad_output_function,
    "mlm": mlm_output_function,
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
