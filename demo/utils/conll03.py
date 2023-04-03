import datasets
from typing import List

_DESCRIPTION = ""


def get_conll(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    tokens = []
    entity_labels = []
    pos_labels = []
    current_sentence = -1

    outs = []
    for line in lines:
        line = line.strip()
        if line:
            if not line.startswith('-DOCSTART-'):
                fields = line.split()
                token = fields[0]
                pos_label = fields[1]
                entity_label = fields[-1]

                tokens.append(token)
                pos_labels.append(pos_label)
                entity_labels.append(entity_label)
        else:
            current_sentence += 1

            if len(tokens) > 0:
                outs.append({
                    "words": tokens,
                    "pos": pos_labels,
                    "labels": entity_labels,
                    "sentence_idx": current_sentence
                })

            tokens = []
            entity_labels = []
            pos_labels = []

    return outs


class Conll03Config(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for the conll03 dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(Conll03Config, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class Conll03Dataset(datasets.GeneratorBasedBuilder):
    """Conll03 dataset"""
    BUILDER_CONFIG_CLASS = Conll03Config
    BUILDER_CONFIGS = [
        Conll03Config(
            name="default",
            description=_DESCRIPTION,
            data_dir="./nlptest/"
        )

    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "words": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['O', 'B-ORG', 'B-LOC', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'I-LOC', 'I-ORG']
                        )
                    ),
                    "pos": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=['NNP', ':', 'IN', 'NN', '.', 'CD', 'VBD', 'DT', 'CC', 'NNS', 'TO',
                                   'VB', 'PRP$', ',', 'RB', 'MD', 'JJ', 'RP', 'VBN', 'VBG', 'VBZ',
                                   'VBP', 'PRP', 'POS', 'WP', 'WDT', '(', ')', 'SYM', 'NNPS', 'JJS',
                                   '$', 'WRB', '"', 'EX', 'WP$', "''", 'RBR', 'RBS', 'FW', 'JJR',
                                   'UH', 'PDT', 'LS', 'NN|SYM']
                        )
                    ),
                    "sentence_idx": datasets.Value("int16"),
                }
            ),
            supervised_keys=None
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": self.config.data_dir + "conll03.conll",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": downloaded_files["train"]
                }
            )
        ]

    def _generate_examples(self, filepath):
        """ Yields examples. """
        data = get_conll(filepath)

        for idx, row in enumerate(data):
            yield idx, row
