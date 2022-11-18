from ipywidgets import HBox, GridspecLayout, AppLayout, Layout, HTML
import ipywidgets as widgets
from IPython.display import clear_output
from typing import List, Dict, Tuple, Optional
from .utils import stylesheet
from pandas import DataFrame


def conll_reader(conll_path: str) -> List[tuple]:
    """
    Read CoNLL file and convert it to the list of labels and sentences.
    :param conll_path: CoNLL file path.
    :return: data and labels which have sentences and labels joined with the single space.
    """
    with open(conll_path) as f:
        data = []
        content = f.read()
        docs = [i.strip() for i in content.strip().split('-DOCSTART- -X- -X- O') if i != '']
        for doc in docs:
            doc_sent = []
            pos_tags = []
            chunk_tags = []
            labels = []

            #  file content to sentence split
            sentences = doc.strip().split('\n\n')

            if sentences == ['']:
                data.append(([''], [''], ['']))
                continue

            for sent in sentences:
                sentence_data = []
                sentence_labels = []
                sentence_pos_tags = []
                sentence_chunk_tags = []

                #  sentence string to token level split
                tokens = sent.strip().split('\n')

                # get annotations from token level split
                token_list = [t.split() for t in tokens]

                #  get token and labels from the split
                for split in token_list:
                    sentence_data.append(split[0])
                    sentence_labels.append(split[-1])
                    sentence_pos_tags.append(split[1])
                    sentence_chunk_tags.append(split[2])

                doc_sent.append(" ".join(sentence_data))
                labels.append(" ".join(sentence_labels))
                pos_tags.append(" ".join(sentence_pos_tags))
                chunk_tags.append(" ".join(sentence_chunk_tags))

            data.append((doc_sent, pos_tags, chunk_tags, labels))

    return data


def conll_writer(
        sentences: List[str],
        pos_tags: List[str],
        chunk_tags: List[str],
        labels: List[str],
        save_path: str,
        docs_indx: Optional[List[int]] = None
) -> int:
    """
    Takes sentences_list, tags_list and labels_list and write to CoNLL file, docs can be separated with docs_indx.
    :param sentences: List of sentences, tokens are joined with whitespace.
    :param pos_tags: List of pos tags.
    :param chunk_tags: List of chunk tags.
    :param labels: List of labels.
    :param docs_indx: List of integers that keep track of documents in CoNLL file.
    :param save_path: CoNLL file path to save.
    :return: number of sentences written to the file.
    """

    try:
        with open(save_path, 'w') as f:
            try:
                counter = 0
                f.write("-DOCSTART- -X- -X- O\n")
                for indx, (sentence, sent_pos, sent_chunk_tag, sent_labels) in enumerate(
                        zip(sentences, pos_tags, chunk_tags, labels)):

                    if docs_indx and counter < len(docs_indx) and indx == docs_indx[counter]:
                        f.write("\n-DOCSTART- -X- -X- O\n")
                        counter += 1

                    f.write("\n")
                    for token, pos_tag, chunk_tag, label in zip(
                            sentence.split(), sent_pos.split(), sent_chunk_tag.split(), sent_labels.split()):
                        f.write(f"{token} {pos_tag} {chunk_tag} {label}\n")

            except (IOError, OSError) as e:
                print(f"Error while writing to the {save_path}.")
                print(e)
                return None

    except (FileNotFoundError, PermissionError, OSError) as e:
        print(f"Error while opening the {save_path}.")
        print(e)
        return None

    return len(sentences)


def update_with_model_predictions(
        conll_path: str,
        fix_df: DataFrame,
        threshold: float = 0.3,
        save_path: Optional[str] = None
) -> str:
    """
    A method to update CoNLL with label fixes. Takes output dataframe of test_label_errors as input.
    :param conll_path: Path to CoNLL data which will be fixed with replacements.
    :param fix_df: test_label_errors output, dataframe consist of sent_indx, token_indx and correct label.
    :param threshold: minimum error score for replacements.
    :param save_path: Path to CoNLL data to save new data with replacements. If None, new file will be created with a
    new name automatically.
    :return: info string with the name of output file and total number of changes.
    """

    if save_path is None:
        conll_filename = conll_path.split('/')[-1]
        conll_filename = conll_filename.split('.')[0]
        save_path = conll_path.replace(conll_filename, f"{conll_filename}_fixed")

    if not 0 < threshold < 1:
        raise ValueError('Threshold must be between 0 and 1.')

    # Applying filter to get highest probable label errors
    filtered_df = fix_df[fix_df.score > threshold]

    # Getting sentence and token indexes to apply fixes
    label_fixes = {(row.sent_indx, row.token_indx): row.prediction for indx, row in filtered_df.iterrows()}

    num_label_fixes = apply_label_fixes(conll_path, label_fixes, save_path)

    if num_label_fixes == 0:
        return f"No fixes applied! You may decrease threshold."

    return f"Total number of {len(label_fixes)} fixes are made and saved to {save_path}."


def apply_label_fixes(conll_path: str, label_fixes: Dict[Tuple[int, int], str], save_path: str = None) -> int:
    """
    A method to update CoNLL with label fixes. Takes CoNLL data path and dictionary of label fixes as input.
    :param conll_path: Path to CoNLL data which will be fixed with replacements.
    :param label_fixes: Dictionary where sentence and token indexes are mapped to replacements. Keys should be passed
    in tuple format, e.g (sent_index, token index) and values should be corresponding entity to be replaced.
    :param save_path: Path to CoNLL data to save new data with replacements. If None, new file will be created with a
    new name automatically.
    :return: total number of label fixes.
    """
    data = conll_reader(conll_path)

    if save_path is None:
        conll_filename = conll_path.split('/')[-1]
        conll_filename = conll_filename.split('.')[0]
        save_path = conll_path.replace(conll_filename, f"{conll_filename}_fixed")

    #   explode sentences since flag_indxs are based on whole conll file.
    sentences = []
    pos_tags = []
    chunk_tags = []
    labels = []

    counter = 0
    docs_indx = []
    for doc in data:
        #   keep track of the doc ending indxs
        counter += len(doc[0])
        docs_indx.append(counter)

        #   collect all doc sentences in the same list to process at the same time
        sentences.extend(doc[0])
        pos_tags.extend(doc[1])
        chunk_tags.extend(doc[2])
        labels.extend(doc[3])

    for (sent_indx, token_indx), replacement in label_fixes.items():
        sent_labels = labels[sent_indx].split()
        sent_labels[token_indx] = replacement
        labels[sent_indx] = " ".join(sent_labels)

    num_sentence = conll_writer(sentences, pos_tags, chunk_tags, labels, save_path, docs_indx=docs_indx)

    if not num_sentence:
        raise ValueError("Error while writing CoNLL file!")

    return len(label_fixes)


def add_flag_to_conll(
        conll_path: str,
        flag_indexes: List[Tuple[int, int]],
        flag: Optional[str] = "*",
        save_path: Optional[str] = None
) -> str:
    """
    A method to add label flag to CoNLL data. Takes CoNLL data path and dictionary of flag fixes as input.
    :param conll_path: Path to CoNLL data.
    :param flag_indexes: List of tuples where each tuple have sent_indx and token_indx, e.g (sent_indx, token_indx)
    :param flag: String that will be replaced with part of speech and syntactic tag in the CoNLL file. Default is *.
    :param save_path: Path to CoNLL data to save new data. If None, new file will be created with a new name
    automatically.
    :return: info string with the name of output file and total number of changes.
    """

    if " " in flag:
        raise ValueError(f" {flag} is invalid! flag cannot contains spaces.")

    data = conll_reader(conll_path)

    if save_path is None:
        conll_filename = conll_path.split('/')[-1]
        conll_filename = conll_filename.split('.')[0]
        save_path = conll_path.replace(conll_filename, f"{conll_filename}_flagged")

    #   explode sentences since flag_indxs are based on whole conll file.
    sentences = []
    pos_tags = []
    chunk_tags = []
    labels = []

    counter = 0
    docs_indx = []
    for doc in data:
        #   keep track of the doc ending indxs
        counter += len(doc[0])
        docs_indx.append(counter)

        #   collect all doc sentences in the same list to process at the same time
        sentences.extend(doc[0])
        pos_tags.extend(doc[1])
        chunk_tags.extend(doc[2])
        labels.extend(doc[3])

    for sent_indx, token_indx in flag_indexes:

        sample_pos_tags = pos_tags[sent_indx].split()
        sample_chunk_tags = chunk_tags[sent_indx].split()

        new_pos_tag = [flag if indx == token_indx else tag for indx, tag in enumerate(sample_pos_tags)]
        new_chunk_tag = [flag if indx == token_indx else tag for indx, tag in enumerate(sample_chunk_tags)]

        pos_tags[sent_indx] = " ".join(new_pos_tag)
        chunk_tags[sent_indx] = " ".join(new_chunk_tag)

    num_sentence = conll_writer(sentences, pos_tags, chunk_tags, labels, save_path, docs_indx=docs_indx)

    if not num_sentence:
        raise ValueError("Error while writing CoNLL file!")

    return f"Total number of {len(flag_indexes)} flags are added and saved to {save_path}."


class InteractiveFix:
    """
    Creates UI in Jupyter Notebook cell for interactive fixing.
    Parameters
    ----------
    result_df: :class:`pandas.DataFrame`
        Pandas DataFrame returned from `find_label_errors`
    default_strategy: str, Default strategy to apply each issue in the page. Can be one of 'flag issue', 'prediction',
    'None'. Default is None.
    page_size: int, Number of sample displayed in each page.
    """

    def __init__(
            self,
            result_df: DataFrame,
            default_strategy: Optional[str] = None,
            page_size: int = 5
    ):

        self.style = f"""
        <style> {stylesheet} </style>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">
        """

        self.table = result_df
        self.label_fixes = dict()

        indx = 2
        ner_classes = dict()
        for label in self.table['ground_truth']:
            if not ner_classes.get(label, None):
                ner_classes[label] = indx
                indx += 1
        self.ner_classes = ner_classes

        self.curr_indx = 0
        self.page_size = page_size
        self.total_error = len(self.table)
        self.default_strategy = default_strategy

    def create_grid(self):

        grid = GridspecLayout(self.page_size, 30, height="500px")
        grid.add_class('ignore_margin')
        return grid

    def fill_grid(
            self,
            grid: GridspecLayout,
    ):

        for i in range(0, self.page_size):
            css_class = 'row_even'
            if i % 2 == 0:
                css_class = 'row_odd'

            grid[i:i + 1, 0:8] = self.get_sentence(self.curr_indx + i, css_class)
            grid[i:i + 1, 8:11] = self.get_cell_element(self.table['chunk'][self.curr_indx + i], css_class)
            grid[i:i + 1, 11:14] = self.get_cell_element(self.table['token'][self.curr_indx + i], css_class)
            grid[i:i + 1, 14:17] = self.get_cell_element(self.table['ground_truth'][self.curr_indx + i], css_class)
            grid[i:i + 1, 17:20] = self.get_cell_element(self.table['prediction'][self.curr_indx + i], css_class)
            grid[i:i + 1, 20:23] = self.get_cell_element('{:.2f}'.format(self.table['prediction_confidence'][self.curr_indx + i]), css_class)
            grid[i:i + 1, 23:26] = self.get_cell_element('{:.2f}'.format(self.table['score'][self.curr_indx + i]), css_class)
            grid[i:i + 1, 26:30] = self.get_drop_down(self.curr_indx + i, css_class)

    def get_header(self):

        head_grid = GridspecLayout(1, 30)
        head_grid[0, 0:8] = self.get_title_html('Sentence')
        head_grid[0, 8:11] = self.get_title_html('Chunk')
        head_grid[0, 11:14] = self.get_title_html('Token')
        head_grid[0, 14:17] = self.get_title_html('Ground Truth')
        head_grid[0, 17:20] = self.get_title_html('Prediction')
        head_grid[0, 20:23] = self.get_title_html('Model Confidence')
        head_grid[0, 23:26] = self.get_title_html('Error Score')
        head_grid[0, 26:30] = self.get_drop_down_title()
        return head_grid

    def get_central_grid(self):

        grid = self.create_grid()
        self.fill_grid(grid)
        return grid

    def get_footer(self):

        grid = GridspecLayout(1, 30)
        grid[0, 0:10] = self.num_sample_drop_down()
        grid[0, 10:13] = self.empty_grid()
        grid[0, 13:18] = self.get_page_number()
        grid[0, 18:25] = self.empty_grid()
        grid[0, 25:28] = self.get_page_control_buttons()
        grid[0, 28:30] = self.empty_grid()
        return grid

    def get_page_control_buttons(self):

        page_left = widgets.Button(
            disabled=False,
            button_style='',
            tooltip='',
            icon='angle-left',
        )
        page_left.add_class('page_button')
        page_left.on_click(self.prev_page)

        page_right = widgets.Button(
            disabled=False,
            button_style='',
            tooltip='',
            icon='angle-right',
        )
        page_right.add_class('page_button')
        page_right.on_click(self.next_page)

        box = HBox(
            children=[
                page_left,
                widgets.HTML(
                    value=
                    f"""<div style='text-align:center; color:#FFFFFF;'>
                    <p> Page <p>
                     </div>""",
                    layout=Layout(height='auto', width='auto')
                ),
                page_right
            ],
            layout=Layout(
                display='flex',
                justify_content="center",
                align_items='center'
            )
        )
        box.add_class('footer')
        return box

    def next_page(self, b):

        if self.curr_indx + self.page_size <= self.total_error:
            self.curr_indx += self.page_size
            self.display()

    def prev_page(self, b):

        if self.curr_indx - self.page_size >= 0:
            self.curr_indx -= self.page_size
            self.display()

    def adjust_page_size(self, value):

        if self.page_size == value.new:
            return None
        self.page_size = value.new
        self.display()

    def num_sample_drop_down(self):

        dropdown = widgets.Dropdown(
            options=[5, 10, 15, 20, 25],
            value=self.page_size,
            layout=Layout(
                width='initial'
            )
        )
        dropdown.observe(self.adjust_page_size, names='value')
        dropdown.add_class('custom-dropdown-white')

        box = HBox(
            children=[
                widgets.HTML(
                    value=f"""<div style='text-align:center; color:#FFFFFF;'>
                    <p> view <p>
                    </div>""",
                    layout=Layout(height='auto', width='auto')
                ),
                dropdown,
                widgets.HTML(
                    value=f"""<div style='text-align:center; color:#FFFFFF;'>
                    <p> per page <p>
                    </div>""",
                    layout=Layout(height='auto', width='auto')
                )
            ],
            layout=Layout(
                display='flex',
                justify_content="center",
                align_items='center'
            )
        )
        box.add_class('footer')
        return box

    def get_page_number(self):

        box = HBox(
            children=[
                widgets.HTML(
                    value=f"""<div style='text-align:center; color:#FFFFFF;'>
            <p> showing {self.curr_indx + 1} / {self.curr_indx + self.page_size} of {self.total_error} <p>
            </div>""",
                    layout=Layout(height='auto', width='auto')
                )
            ],
            layout=Layout(
                display='flex',
                justify_content="center",
                align_items="center"
            )
        )
        box.add_class('footer')
        return box

    def empty_grid(self):

        empty_html = widgets.HTML(
            value=f"""<div>
            </div>""",
            layout=Layout(
                height='auto',
                width='auto',
                justify_content="center",
                align_items='center'
            ),
        )
        empty_html.add_class('footer')
        return empty_html

    def get_classes(self):
        return self.ner_classes

    def fix_label(self, value):
        indx = value.new
        replace_by, (sent_indx, token_indx) = value['owner'].options[indx]
        self.label_fixes[(sent_indx, token_indx)] = replace_by

    def apply_fixes(self, conll_path: str, save_path: str = None):

        replacements = dict()
        flags = []

        for k, v in self.label_fixes.items():

            if v is None:
                continue

            elif v == 'flag issue':
                flags.append(k)

            else:
                replacements[k] = v

        if save_path is None:
            conll_filename = conll_path.split('/')[-1]
            conll_filename = conll_filename.split('.')[0]
            save_path = conll_path.replace(conll_filename, f"{conll_filename}_fixed")

        if not replacements:
            print('No label fixes are passed. Skipping...')
        else:
            num_replacement = apply_label_fixes(conll_path, replacements, save_path)
            print(f"Total number of {num_replacement} replacements are made and saved to {save_path}")
            conll_path = save_path

        if not flags:
            print('No flags indexes are passed. Skipping...')
        else:
            print(add_flag_to_conll(conll_path, flags, save_path))

    def display(self):
        clear_output(wait=True)
        display(HTML(self.style))
        display(
            AppLayout(
                header=self.get_header(),
                center=self.get_central_grid(),
                footer=self.get_footer(),
                pane_heights=['70px', '500px', '70px'],
                grid_gap="0px"
            )
        )

    def fix_all(self, strategy):
        self.default_strategy = strategy.new
        self.display()

    def get_drop_down_title(self):
        dropdown = widgets.Dropdown(
            options=[None, 'prediction', 'flag issue'],
            value=self.default_strategy,
            layout=Layout(
                width='initial'
            )
        )
        dropdown.observe(self.fix_all, names='value')
        dropdown.add_class('custom-dropdown-white')

        box = HBox(
            children=[
                dropdown
            ],
            layout=Layout(
                display='flex',
                justify_content="center",
                align_items='center',
                padding='15px'
            )
        )
        box.add_class('header')

        return box

    def get_cell_element(self, element, css_class):
        box = HBox(
            children=[
                widgets.HTML(
                    value=f"""<div style='text-align:center; color:#262626;'>
                <p> {element} <p>
                </div>""",
                    layout=Layout(height='auto', width='auto')
                )
            ],
            layout=Layout(
                display='flex',
                justify_content="center",
                align_items="center"
            )
        )
        box.add_class(css_class)
        return box

    def get_sentence(self, x, css_class):
        widget = widgets.HTML(
            value=f"""<div style='text-align:left;'>
            <p> {self.table['sentence'][x]} <p>
            </div>""",
            layout=Layout(height='auto', width='auto', padding='10px')
        )
        widget.add_class(css_class)
        return widget

    def get_default_indx(self, indx):

        if self.default_strategy == 'prediction':
            prediction = self.table['prediction'][indx]
            return self.ner_classes[prediction]

        elif self.default_strategy == 'flag issue':
            return 1

        else:
            return 0

    def get_drop_down(self, indx, css_class):

        indx_tuple = (self.table['sent_indx'][indx], self.table['token_indx'][indx])

        options = [(k, indx_tuple) for k in self.ner_classes]
        options.insert(0, (None, indx_tuple))
        options.insert(1, ('flag issue', indx_tuple))

        default_indx = self.get_default_indx(indx)
        self.label_fixes[indx_tuple] = options[default_indx][0]

        dropdown = widgets.Dropdown(
            options=options,
            index=default_indx,
            layout=Layout(
                width='initial'
            )
        )
        dropdown.observe(self.fix_label, names=['value', 'index'])
        dropdown.add_class('custom-dropdown-black')

        box = HBox(
            children=[
                dropdown
            ],
            layout=Layout(
                display='flex',
                justify_content="center",
                align_items='center',
                padding='15px'
            )
        )
        box.add_class(css_class)
        return box

    def get_title_html(self, string):

        title_html = widgets.HTML(
            value=f"""
            <div class="table_title">
            <span class="title_text"> {string} </span>
            </div>""",
        )
        title_html.add_class('header')
        return title_html