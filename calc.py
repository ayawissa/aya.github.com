import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox
import numpy as nppi
from PyQt5.QtCore import Qt


class LinearSystemSolver(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Système linéaire ')
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        # Welcome Header
        welcome_label = QLabel('BY Djouab Aya Wissam - Groupe 2', self)
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 16px; color: #333; font-weight: bold;")
        self.layout.addWidget(welcome_label)

        # Rest of the UI components
        self.label_matrix = QLabel('Entrez la matrice de coefficients (valeurs séparées par des virgules, lignes séparées par un point-virgule):') 
        self.entry_matrix = QLineEdit(self)
        self.layout.addWidget(self.label_matrix)
        self.layout.addWidget(self.entry_matrix)

        self.label_vector = QLabel('Entrez le vecteur constant (valeurs séparées par des virgules):')
        self.entry_vector = QLineEdit(self)
        self.layout.addWidget(self.label_vector)
        self.layout.addWidget(self.entry_vector)

        self.method_dropdown = QComboBox(self)
        self.method_dropdown.addItem('Gauss')
        self.method_dropdown.addItem('Jacobi')
        self.method_dropdown.addItem('Gauss-Seidel')
        self.layout.addWidget(self.method_dropdown)

        self.solve_button = QPushButton('Calculer', self)
        self.solve_button.clicked.connect(self.solve_linear_system)
        self.layout.addWidget(self.solve_button)

        self.result_label = QLabel('Résultat:', self)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    # ... (rest of the methods remain unchanged)


    def solve_linear_system(self):
        try:
            matrix_str = self.entry_matrix.text()
            vector_str = self.entry_vector.text()

            matrix = np.array([list(map(float, row.split(','))) for row in matrix_str.split(';')])
            vector = np.array(list(map(float, vector_str.split(','))))

            method = self.method_dropdown.currentText()

            if method == 'Gauss':
                result = np.linalg.solve(matrix, vector)
            elif method == 'Jacobi':
                initial_guess = np.zeros_like(vector)
                result = self.jacobi_method(matrix, vector, initial_guess)
            elif method == 'Gauss-Seidel':
                initial_guess = np.zeros_like(vector)
                result = self.gauss_seidel_method(matrix, vector, initial_guess)
            else:
                raise ValueError("Invalid method selected.")

            self.result_label.setText(f"Result: {result}")
        except Exception as e:
            self.result_label.setText(f"Ereur: {e}")

    def jacobi_method(self, A, b, initial_guess, tolerance=1e-10, max_iterations=100):
        x = initial_guess.copy()
        D = np.diag(np.diag(A))
        LU = A - D
        for _ in range(max_iterations):
            x_new = np.linalg.inv(D).dot(b - LU.dot(x))
            if np.linalg.norm(x_new - x) < tolerance:
                return x_new
            x = x_new
        raise Exception("La méthode Jacobi n'a pas convergé dans la tolérance et les itérations spécifiées.")

    def gauss_seidel_method(self, A, b, initial_guess, tolerance=1e-10, max_iterations=100):
        x = initial_guess.copy()
        for _ in range(max_iterations):
            for i in range(len(x)):
                x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
            if np.linalg.norm(A.dot(x) - b) < tolerance:
                return x
        raise Exception("La méthode de Gauss-Seidel n’a pas convergé dans les limites de tolérance et d’itérations spécifiées.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LinearSystemSolver()
    window.show()
    sys.exit(app.exec_())
