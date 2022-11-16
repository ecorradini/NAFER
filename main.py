import sys
from PyQt6.QtWidgets import QApplication

from ui.welcomeview import WelcomeView

if __name__ == '__main__':
    app = QApplication(sys.argv)
    welcome_view = WelcomeView()
    welcome_view.show()
    sys.exit(app.exec())
