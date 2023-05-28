import logging

from .MLM.PlugDPLFormatter import PlugDPLFormatter


from .TextClassification.TextClassificationPlugDFormatter import TextClassificationPlugDFormatter
from .TextClassification.TextClassificationAdapterFormatter import TextClassificationAdapterFormatter

from .FEVER.FEVERFormatter import FEVERFormatter
from .FEVER.FEVERCtxFormatter import FEVERCtxFormatter,FEVERCtxPlugDFormatter,FEVERCtxED2LMFormatter

from .OpenQA.OpenQAFormatter import OpenQAFormatter,OpenQAPlugDFormatter


logger = logging.getLogger(__name__)

formatter_list = {
    "None": lambda x: None,

    "TextClassificationPlugD": TextClassificationPlugDFormatter,
    "TextClassificationAdapter": TextClassificationAdapterFormatter,

    "FEVER": FEVERFormatter,
    "FEVERCtx": FEVERCtxFormatter,
    "FEVERPlugD": FEVERCtxPlugDFormatter,

    "OpenQA": OpenQAFormatter,
    "OpenQAPlugD": OpenQAPlugDFormatter,

    "PlugDPL": PlugDPLFormatter,
}

def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)
    print("formatter_type", which)
    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError
