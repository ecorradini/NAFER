import shutil
from os import listdir
from os.path import isfile, join
import pandas as pd
from PyQt6.QtCore import QUrl

from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6 import uic, QtWidgets
from PyQt6.QtWidgets import QWidget, QFileDialog

from ui.resultview import ResultView
from utilities.pandas_model import PandasModel


class WelcomeView(QWidget):

    def __init__(self, parent=None):
        super(WelcomeView, self).__init__(parent)
        uic.loadUi('ui/welcome_view.ui', self)
        self.current_dataset = None
        self.current_name = None
        self._fill_default_list()

        self.default_list.selectionModel().selectionChanged.connect(self._default_selection_changed)

        self.custom_button.clicked.connect(self._select_csv_dialog)

        self.next_button.clicked.connect(self._go_next)

    def _get_default_csvs(self):
        files_1 = [f for f in listdir('default_datasets') if isfile(join('default_datasets', f))]
        files_2 = [f for f in listdir('custom_datasets') if isfile(join('custom_datasets', f))]
        return files_1 + files_2

    def _fill_default_list(self):
        self.csvs = self._get_default_csvs()
        model = QStandardItemModel(self.default_list)
        for csv in self.csvs:
            item = QStandardItem(csv)
            font = item.font()
            font.setPointSize(18)
            item.setFont(font)
            model.appendRow(item)
        self.default_list.setModel(model)

    def _default_selection_changed(self):
        selected_names = [self.csvs[idx.row()] for idx in self.default_list.selectedIndexes()]
        self._fill_table(selected_names[0])

    def _fill_table(self, dataset_name):
        try:
            data = pd.read_csv(f'default_datasets/{dataset_name}', header=0)
        except FileNotFoundError:
            data = pd.read_csv(f'custom_datasets/{dataset_name}', header=0)
        self.model = PandasModel(data.describe())
        self.table_default_properties.setModel(self.model)
        self.current_dataset = data
        self.current_name = dataset_name

    def _select_csv_dialog(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Select CSV",
            "${HOME}",
            "CSV Files (*.csv)",
        )
        url = QUrl.fromLocalFile(fname[0])
        if len(fname[0]) > 0:
            shutil.copyfile(fname[0], f'custom_datasets/{url.fileName()}')
        self._fill_default_list()
        data = pd.read_csv(fname[0], header=0)
        self.model = PandasModel(data.describe())
        self.table_custom_properties.setModel(self.model)
        self.current_dataset = data
        self.current_name = url.fileName()

    def _go_next(self):
        if self.current_dataset is not None:
            self.result_view = ResultView(data=self.current_dataset, name=self.current_name)
            self.result_view.show()
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Please, select or add a dataset!')
            error_dialog.exec()


