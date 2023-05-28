from .kara.KaraDataset import make_kara_dataset


from .NQ.NQDataset import NQDataset
from .Json.JsonDataset import JsonDataset
from .Json.JsonlineDataset import JsonlineDataset

from .FEVER.FEVERDataset import FEVERDataset

from .OpenQA.OpenQADataset import OpenQADataset,FewOpenQADataset
from .OpenQA.SQuADDataset import SQuADDataset,FewSQuADDataset
from .OpenQA.OpenQADataset2 import OpenQADataset2
from .OpenQA.SFDataset import SFDataset


dataset_list = {
    "NQT5": NQDataset,

    "json": JsonDataset,
    "json-line": JsonlineDataset,

    "kara": make_kara_dataset,

    "FEVER": FEVERDataset,
    "OpenQA": OpenQADataset,
    "SQuAD": SQuADDataset,
    "OpenQA2": OpenQADataset2,
    "SlotFilling": SFDataset,
    "FewSQuAD": FewSQuADDataset,
    "FewOpenQA": FewOpenQADataset,
}
