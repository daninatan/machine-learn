import tkinter as tk
from tkinter import ttk

class ModelSelectorUI:
    def __init__(self):
        self.selected_models = []

    def run(self):
        self.root = tk.Tk()
        self.root.title("GawLL")
        self.root.geometry("350x300")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f6fa")

        style = ttk.Style()
        style.theme_use("clam")

        # estilos customizados
        style.configure("TFrame", background="#f5f6fa")
        style.configure("TLabel", background="#f5f6fa", font=("Segoe UI", 11))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("TCheckbutton", background="#f5f6fa", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True)

        # título
        ttk.Label(
            main_frame,
            text="Selecionar Modelos",
            style="Title.TLabel"
        ).pack(pady=(0, 15))

        # checkboxes
        self.vars = {
            "knn": tk.BooleanVar(),
            "dt": tk.BooleanVar(),
            "rf": tk.BooleanVar(),
            "mlp": tk.BooleanVar()
        }

        checkbox_frame = ttk.Frame(main_frame)
        checkbox_frame.pack()

        for model, var in self.vars.items():
            ttk.Checkbutton(
                checkbox_frame,
                text=model.upper(),
                variable=var
            ).pack(anchor="w", pady=3)

        # botão
        ttk.Button(
            main_frame,
            text="Executar",
            command=self.submit
        ).pack(pady=20, ipadx=10, ipady=5)

        self.root.mainloop()
        return self.selected_models

    def submit(self):
        self.selected_models = [
            model for model, var in self.vars.items() if var.get()
        ]
        self.root.destroy()