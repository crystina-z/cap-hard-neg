from capreolus import ModuleBase, Dependency, ConfigOption, constants
from capreolus.extractor import Extractor, BertPassage


@Extractor.register
class LCEBertPassage(BertPassage):
    module_name = "LCEbertpassage"

    def id2vec(self,):
        # TODO
        pass
