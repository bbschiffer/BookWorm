import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import motorControl
import pandas as pd, time

# Dictionary to store book locations
book_locations = {}

# def setup_styles():
#     global style
#     style = ttk.Style()
#     style.configure(
#         "Custom.TButton",
#         font=("Arial", 12, "bold"),
#         foreground="white",
#         background="#0ac499",
#         padding=8,
#         relief="flat"
#     )
#     style.map(
#         "Custom.TButton",
#         background=[
#             ("active", "#089e78"),   # hover color
#             ("pressed", "#7a1306")   # click color
#         ],
#         foreground=[
#             ("active", "black"),     # hover text color
#             ("pressed", "white")     # click text color
#         ]
#     )

def show_main_menu():
    clear_frame()
    tk.Label(root, bg='white', text="Welcome to BookWorm!", font=("Arial",22)).pack(pady=12)

    tk.Button(root, text="Add New Book",highlightbackground="#ffffff" , command=show_add_book).pack(pady=10)
    tk.Button(root, text="Find Book Location", highlightbackground="#ffffff", command=show_find_book).pack(pady=10)
    tk.Button(root, text="Quit", width=2,highlightbackground="#ffffff", command=root.destroy).pack(padx=5,side= "left")

def show_add_book():
    clear_frame()
    tk.Label(root, text="Add a New Book", font=("Arial", 18)).pack(pady=10)

    tk.Label(root, text="Book Title:",font=("Arial", 12)).pack()
    title_entry = tk.Entry(root, width=30)
    title_entry.pack(pady=5)

    tk.Label(root, text="Cubby Location:",font=("Arial", 12)).pack()
    cubby_entry = tk.Entry(root, width=30)
    cubby_entry.pack(pady=5)

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
    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, anchor='center') 
    tk.Button(root, text="Save", command=save_book).pack()
    tk.Button(root, text="Back", width=2, command=show_main_menu).pack(side= tk.LEFT, pady=10)

def show_find_book():
    clear_frame()
    tk.Label(root, text="Find Book Location", font=("Arial", 18)).pack(pady=10)

    tk.Label(root, text="Book Title:",font=("Arial", 12)).pack(pady = 5)
    title_entry = tk.Entry(root, width=30)
    title_entry.pack()

    def find_book():
        title = title_entry.get().strip()
        if not title:
            messagebox.showerror("Error", "Please enter a book title!")
            return
        elif title in book_locations:
            cubby = book_locations[title]
            messagebox.showinfo("Book Found", f" {title} is located in cubby {cubby}.")
        else:
            messagebox.showerror("Not Found", f" {title} not found in the database.")
    
    def retrieve_book():
        title = title_entry.get().strip()
        if not title:
            messagebox.showerror("Error", "Please enter a book title!")
            return
        elif title in book_locations:
        
            #del book_locations[title]
            
            cubby = book_locations[title]
            x_cubby = get_location(cubby)
            motorControl.moveMotor(x_cubby,.2)
            print(" ")
            
            print("End effector retrieving container")
            time.sleep(1) #Sleep command to represent end effector moving
            print(" ")
            
            x_dropoff = get_location("Dropoff")
            distance = x_dropoff - x_cubby
            motorControl.moveMotor(distance,.2);
            print(" ")
            
            print("End effector depositing container")
            time.sleep(1) #Sleep command to represent end effector moving
            print(" ")
            
            messagebox.showinfo("Book retrieved", "Please take book from cubby")
        else:
            messagebox.showinfo("Not found", "Book not found")
            
            show_done_menu()

    button_frame = tk.Frame(root)
    button_frame.pack(expand=True, anchor='center') 
    tk.Button(button_frame, text="Search", command=find_book).pack(side= tk.LEFT)
    tk.Button(button_frame, text="Retrieve", command=retrieve_book).pack(side= tk.LEFT) # REPLACE FIND_BOOK WITH RETRIEVE_BOOK
    tk.Button(root, text="Back", width=2, command=show_main_menu).pack(side= tk.LEFT, pady=10)

def show_done_menu():
    clear_frame()
    tk.Label(root, text="Return Cubby?", font=("Arial", 18)).pack(pady=10)

    tk.Label(root, text="(Y/N)",font=("Arial", 12)).pack(pady = 5)
    return_entry = tk.Entry(root, width=30)
    return_entry.pack()

def get_location(location):
    ldf = pd.read_csv('LocationDatabase.csv')
    
    # Filter the DataFrame
    result = ldf[ldf["Location"].str.contains(str(location))]
    
    xCoord = result.iloc[0]["X"]
    #yCoord = result.iloc[0]["Y"]
    return xCoord
    

def clear_frame():
    for widget in root.winfo_children():
        widget.destroy()

# Main window
root = tk.Tk()
root.title("BookWorm")
root.geometry("400x250")
root.config(bg="white")

show_main_menu()
root.mainloop()
