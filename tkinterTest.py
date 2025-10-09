from tkinter import *
from tkinter import ttk

root = Tk()
root.geometry("300x100")  # width x height

# frm = ttk.Frame(root, padding=10)
# frm.grid()
ttk.Label(root, text="Hello World!").grid(column=0, row=0)
ttk.Button(root, text="Quit", command=root.destroy).grid(column=1, row=0)

root.mainloop()
