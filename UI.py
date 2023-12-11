import tkinter as tk
from tkinter import filedialog


def SelecDirectory():
    # Create a root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file selection dialog
    dirPath = filedialog.askdirectory()

    # Print the selected file path
    print(f"Selected directory: {dirPath}")

    # Close the root window
    root.destroy()

    return dirPath