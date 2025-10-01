import tkinter as tk
from tkinter import messagebox

# Dictionary to store book locations
book_locations = {}

def show_main_menu():
    clear_frame()
    tk.Label(root, text="Welcome to BookWorm!", font=("Arial", 16)).pack(pady=20)

    tk.Button(root, text="Add New Book", width=20, command=show_add_book).pack(pady=10)
    tk.Button(root, text="Find Book Location", width=20, command=show_find_book).pack(pady=10)

def show_add_book():
    clear_frame()
    tk.Label(root, text="Add a New Book", font=("Arial", 14)).pack(pady=10)

    tk.Label(root, text="Book Title:").pack()
    title_entry = tk.Entry(root, width=30)
    title_entry.pack()

    tk.Label(root, text="Cubby Location:").pack()
    cubby_entry = tk.Entry(root, width=30)
    cubby_entry.pack()

    def save_book():
        title = title_entry.get().strip()
        cubby = cubby_entry.get().strip()

        if not title or not cubby:
            messagebox.showerror("Error", "Please enter both title and location!")
            return
        if not cubby.isdigit():
            messagebox.showerror("Error", "Cubby location must be numeric!")
            return

        book_locations[title] = cubby
        messagebox.showinfo("Success", f"{title} saved in cubby {cubby}")
        show_main_menu()

    tk.Button(root, text="Save", command=save_book).pack(pady=10)
    tk.Button(root, text="Back", command=show_main_menu).pack()

def show_find_book():
    clear_frame()
    tk.Label(root, text="Find Book Location", font=("Arial", 14)).pack(pady=10)

    tk.Label(root, text="Book Title:").pack()
    title_entry = tk.Entry(root, width=30)
    title_entry.pack()

    def find_book():
        title = title_entry.get().strip()
        if title in book_locations:
            cubby = book_locations[title]
            messagebox.showinfo("Book Found", f" {title} is located in cubby {cubby}.")
        else:
            messagebox.showerror("Not Found", f" {title} not found in the database.")

    tk.Button(root, text="Search", command=find_book).pack(pady=10)
    tk.Button(root, text="Back", command=show_main_menu).pack()

def clear_frame():
    for widget in root.winfo_children():
        widget.destroy()

# Main window
root = tk.Tk()
root.title("BookWorm")
root.geometry("300x300")

show_main_menu()
root.mainloop()
