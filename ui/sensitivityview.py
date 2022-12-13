from PyQt6 import uic
from PyQt6.QtWidgets import QWidget

from utilities.pandas_model import PandasModel
from utilities.sensitivity import Sensitivity


class SensitivityView(QWidget):

    def __init__(self, data, name, parent=None):
        super(SensitivityView, self).__init__(parent)
        uic.loadUi('ui/sensitivity_view.ui', self)
        self.dataset_name = name
        self.data = data

        self.combo_class.clear()
        self.combo_class.addItems(self.data.columns)
        self.current_class = self.data.columns[0]
        self.combo_class.currentTextChanged.connect(self._on_class_combobox_changed)

        self.models = ["NB", "SVMP", "SVMR", "MLP", "RF"]
        self.combo_models.clear()
        self.combo_models.addItems(self.models)
        self.combo_models.currentTextChanged.connect(self._on_model_combobox_changed)
        self.current_model = self.models[0]

        self.percentage = self.perc_spinbox.value()
        self.perc_spinbox.valueChanged.connect(self._on_spin_valuechange)
        self.sensitivity = Sensitivity(data, self.current_model, self.current_class, self.percentage)

        self.btn_run.clicked.connect(self._run)

    def _on_model_combobox_changed(self, value):
        self.current_model = value

    def _on_class_combobox_changed(self, value):
        self.current_class = value

    def _on_spin_valuechange(self):
        self.percentage = self.perc_spinbox.value()

    def _run(self):
        sensitivity = Sensitivity(self.data,self.current_model,self.current_class,self.percentage)
        sensitivities = sensitivity.compute()
        sensitivities.to_csv(f'sensitivity_{self.dataset_name.replace(".csv", "")}_{self.current_model}_{self.percentage}.csv', index=False)
        self.model2 = PandasModel(sensitivities)
        self.table_sensitivity.setModel(self.model2)

        overall_sensitivities = sensitivity.compute_overall_sensitivity()
        self.model3 = PandasModel(overall_sensitivities)
        self.overall_sensitivity_table.setModel(self.model3)