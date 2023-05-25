from tkinter import *


def setData():
    root = Tk()
    root.geometry('500x500')
    root.title("Stock Market Predictor")

    def getData():
        ss = stocks_symbol.get()
        td = training_data.get()
        op = var.get()
        root.quit()
        return ss, td, op

    stocks_symbol = StringVar()
    training_data = IntVar()

    label_0 = Label(root, text="Stock Market Predictor", width=20, font=("bold", 20))
    label_0.place(x=90, y=53)

    label_1 = Label(root, text="Stock Symbol", width=20, font=("bold", 10))
    label_1.place(x=80, y=130)

    entry_1 = Entry(root, textvar=stocks_symbol)
    entry_1.place(x=240, y=130)

    label_2 = Label(root, text="Training Data(in %)", width=20, font=("bold", 10))
    label_2.place(x=68, y=180)

    entry_2 = Entry(root, textvar=training_data)
    entry_2.place(x=240, y=180)

    label_3 = Label(root, text="Graph output", width=20, font=("bold", 10))
    label_3.place(x=70, y=230)
    var = IntVar()
    Radiobutton(root, text="Yes", padx=5, variable=var, value=1).place(x=235, y=230)
    Radiobutton(root, text="No", padx=20, variable=var, value=2).place(x=290, y=230)

    Button(root, text='Submit', width=20, bg='brown', fg='white', command=getData).place(x=180, y=380)
    root.mainloop()
    root.destroy()
    stocks_symbol, training_data, output_graph = getData()
    return stocks_symbol, training_data, output_graph


def popupmsg(output):
    popup = Tk()
    popup.wm_title("Output")
    label = Label(popup, text="Accuracy: " + str(output * 100) + "%")
    label.pack(side="top", fill="x", padx=100, pady=100)
    b1 = Button(popup, text="Okay", command=popup.destroy)
    b1.pack()
    popup.mainloop()



