import tkinter as tk
from tkinter import ttk
import numpy as np

class MatrixViewer:
    def __init__(self, matrix, title="Matrix Viewer"):
        self.matrix = matrix
        self.rows, self.cols = matrix.shape
        self.highlight_color = "red"
        self.title = title
        
        self.font_size = 10

    def visualize(self):
        root = tk.Tk()
        root.title(self.title)
        
        tree = ttk.Treeview(root)
        tree["columns"] = list(range(self.cols))
        
        for col in range(self.cols):
            tree.heading(col, text=f"Column {col}")
        
        for i in range(self.rows):
            row_values = [f"{self.matrix[i][j].real}+{self.matrix[i][j].imag}j" for j in range(self.cols)]
            tree.insert("", "end", text=f"Row {i}", values=row_values)
        
        tree.pack(fill="both", expand=True)

        # Add horizontal scrollbar
        h_scrollbar = tk.Scrollbar(root, orient="horizontal", command=tree.xview)
        h_scrollbar.pack(side="bottom", fill="x")
        tree.configure(xscrollcommand=h_scrollbar.set)

        root.mainloop()

    def set_title(self, title):
        self.title = title

def main():

    # Example usage:
    matrix = np.array([[1j if i == j else complex(i, j) for j in range(15)] for i in range(15)])  # Example 15x15 complex matrix
    matrix_viewer = MatrixViewer(matrix, title="My Matrix Viewer")
    matrix_viewer.visualize()

if __name__ == "__main__":
    main()
