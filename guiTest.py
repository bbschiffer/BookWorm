import tkinter as tk
from tkinter import messagebox

allBookLocs = dict()

def submit():
    # Get text from entry boxes
    title = book_entry.get()
    cubby = cubby_entry.get()
    if not title or not cubby:
        messagebox.showwarning("Input Error", "Please fill in both fields.")
        return
    if not cubby.isdigit():
        messagebox.showerror("Invalid Input", "Cubby must be a number.")
        cubby_entry.delete(0, tk.END)  # clear the bad entry
        return
    # Save as global variables (or handle however you like)
    global saved_title, saved_cubby
    saved_title = title
    saved_cubby = cubby

    print("Name entered:", saved_title)
    print("Cubby entered:", saved_cubby)

    root.destroy()  # close the window after
    # add to dictionary
    allBookLocs[saved_title] =  saved_cubby # for later use

def locateBook(title):
    return allBookLocs.get(title, "Book not found")

root = tk.Tk()
root.title("New Book Entry")

# Name label + entry
tk.Label(root, text="Enter Book Title:").pack(padx=10, pady=5)
book_entry = tk.Entry(root, width=30)
book_entry.pack(padx=10, pady=5)

# Age label + entry
tk.Label(root, text="Enter the cubby you placed the book in:").pack(padx=10, pady=5)
cubby_entry = tk.Entry(root, width=30)
cubby_entry.pack(padx=10, pady=5)

# Submit button
tk.Button(root, text="Submit", command=submit).pack(padx=10, pady=10)

root.mainloop()

# After window closes, the inputs are available
print("Saved values -> Name:", saved_title, ", Age:", saved_cubby)
