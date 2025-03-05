
# tkinker
Tkinter is the standard GUI (Graphical User Interface) library for Python. It provides an easy way to create interactive applications with buttons, text fields, labels, menus, and more. Tkinter is built into Python, so it doesn't require separate installation. More details on the [tutorials](https://www.geeksforgeeks.org/python-gui-tkinter/)

Here's a simple Python script that demonstrates the use of tkinter for creating a basic file dialog and showing a message box:
```python

import tkinter as tk  # Import the Tkinter library and alias it as 'tk'
from tkinter import filedialog, messagebox  # Import specific modules from Tkinter for file dialogs and message boxes

def open_file():
    """Function to open a file using a file dialog and display the selected file path in a message box."""
    file_path = filedialog.askopenfilename(  # Open a file dialog to select a file
        title="Select a File",  # Set the dialog title
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]  # Define allowed file types
    )
    if file_path:  # Check if a file was selected (not empty)
        messagebox.showinfo("File Selected", f"You selected:\n{file_path}")  # Show an info message with the file path

def save_file():
    """Function to save a file using a file dialog and write a test message into it."""
    file_path = filedialog.asksaveasfilename(  # Open a save file dialog
        title="Save File",  # Set the dialog title
        defaultextension=".txt",  # Default file extension for saving
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]  # Define allowed file types
    )
    if file_path:  # Check if a file path was provided
        with open(file_path, 'w') as file:  # Open the file in write mode
            file.write("This is a test file.")  # Write some text into the file
        messagebox.showinfo("File Saved", f"File saved at:\n{file_path}")  # Show an info message with the saved file path

# Create the main application window
root = tk.Tk()  # Initialize the main Tkinter window
root.title("Tkinter File Dialog Example")  # Set the window title
root.geometry("300x200")  # Set the window size (width x height)

# Create and place a button to open a file
btn_open = tk.Button(root, text="Open File", command=open_file)  # Create a button to open a file
btn_open.pack(pady=10)  # Place the button in the window with vertical padding

# Create and place a button to save a file
btn_save = tk.Button(root, text="Save File", command=save_file)  # Create a button to save a file
btn_save.pack(pady=10)  # Place the button in the window with vertical padding

# Run the Tkinter event loop to keep the window open
root.mainloop()  # Start the GUI application
```
1. filedialog.askopenfilename() - Opens a file selection dialog.
2. filedialog.asksaveasfilename() - Opens a save file dialog.
3. messagebox.showinfo() - Displays a message box with the selected file path.
4. The GUI consists of two buttons: one for opening a file and another for saving a file.



# Pandas
Pandas is an open-source Python library used for data manipulation, analysis, and preprocessing. It provides powerful data structures such as DataFrame and Series that simplify working with structured data, making it essential for data science, machine learning, and analytics. More details at the [tutorials](https://www.w3schools.com/python/pandas/default.asp)

## Key Features of Pandas
### 1. Data Handling and Manipulation
Provides DataFrame (tabular data, similar to an Excel table) and Series (one-dimensional labeled array) for structured data.
Enables fast data filtering, aggregation, merging, and transformation.

### 2. Reading and Writing Data
Supports reading from and writing to multiple file formats such as:
CSV (pd.read_csv())
Excel (pd.read_excel())
SQL Databases (pd.read_sql())
JSON, Parquet, and more

### 3. Data Cleaning & Preprocessing
Handling missing values (df.dropna(), df.fillna())
Converting data types (pd.to_numeric(), astype())
Duplicated data removal (df.drop_duplicates())

### 4. Data Aggregation & Analysis
Provides built-in statistical methods (df.mean(), df.median(), df.describe())
Grouping and aggregation with groupby()
Pivot tables for summarization (df.pivot_table())

### 5. Merging & Joining Datasets
Similar to SQL operations (merge(), join(), concat())

### 6. Indexing & Selection
Selecting specific rows and columns using .loc[] and .iloc[]
Advanced filtering using boolean indexing

### 7. Integration with Other Libraries
Works well with NumPy, Matplotlib, Seaborn, and Scikit-Learn for visualization, numerical computations, and machine learning.

### Example

```python
import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'Salary': [50000, 60000, 70000]}
df = pd.DataFrame(data)

# Display DataFrame
print(df)

# Convert Age column to numeric
df['Age'] = pd.to_numeric(df['Age'])

# Drop rows with missing values
df_cleaned = df.dropna()

# Reset index after dropping
df_cleaned = df_cleaned.reset_index(drop=True)

# Display cleaned DataFrame
print(df_cleaned)
```
Why Use Pandas?
Efficiency: Handles large datasets quickly.
Ease of Use: Simple, readable syntax similar to SQL and Excel.
Versatility: Used in data science, finance, web scraping, and more.

More example
```python
# Import the pandas library and alias it as 'pd'
import pandas as pd

# Read a CSV file into a Pandas DataFrame
# Assume 'data.csv' is a file with columns: 'ID', 'Name', 'Age', 'Salary'
df = pd.read_csv("data_1.csv")

# Display the first few rows of the DataFrame
print("Original DataFrame:")
print(df.head())

# Convert the 'Age' and 'Salary' columns to numeric values (if not already numeric)
# Errors='coerce' will convert non-numeric values to NaN (Not a Number)
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df["Salary"] = pd.to_numeric(df["Salary"], errors='coerce')

# Display DataFrame after conversion
print("\nDataFrame after converting 'Age' and 'Salary' to numeric:")
print(df.head())

# Drop rows with missing values (NaN)
df_cleaned = df.dropna()

# Display DataFrame after dropping missing values
print("\nDataFrame after dropping rows with missing values:")
print(df_cleaned.head())

# Reset index after dropping rows (to ensure continuous indexing)
df_cleaned = df_cleaned.reset_index(drop=True)

# Display DataFrame after resetting the index
print("\nDataFrame after resetting index:")
print(df_cleaned.head())
```
Please download this [data_1.csv](https://github.com/ph7klw76/Advanced_Computational_Methods/blob/main/data_1.csv) file to run

# Codes that create a user interface

## 1. Import and Global Variables

```python
import tkinter as tk 
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# Global variables to hold the data
theoretical_df = None  # DataFrame for absorption.txt data
experimental_df = None  # DataFrame for experimental.txt data
```
FigureCanvasTkAgg: A Matplotlib utility that embeds a Matplotlib figure into a Tkinter widget.

os: Used for file path manipulations.

The code sets up two global variables (theoretical_df and experimental_df) that will hold the loaded data. They begin as None so the program knows if data has been loaded or not.

The data for theoretical_df can be found [here](https://github.com/ph7klw76/Advanced_Computational_Methods/blob/main/absorption-theory.txt)

The data for experimental_df can be found [here](https://github.com/ph7klw76/Advanced_Computational_Methods/blob/main/experiment.txt)

## 2.Loading Theoretical Data

```python
def load_theoretical_data():
    global theoretical_df
    file_path = filedialog.askopenfilename(title="Select Theoretical Data (absorption.txt)")
    if file_path and os.path.exists(file_path):
        try:
            # Read the theoretical data (assuming two columns: Energy (eV) and OscillatorStrength)
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Energy", "OscillatorStrength"])
            
            # Convert to numeric and drop any NaNs
            df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
            df["OscillatorStrength"] = pd.to_numeric(df["OscillatorStrength"], errors="coerce")
            df.dropna(inplace=True)

            # Convert energy (eV) to wavelength (nm) using λ (nm) = 1239.84 / E (eV)
            df["Wavelength"] = 1239.84 / df["Energy"]

            # Sort by wavelength
            df.sort_values("Wavelength", inplace=True)

            # Save the resulting DataFrame to the global variable
            theoretical_df = df.reset_index(drop=True)

            # Let the user know it succeeded
            messagebox.showinfo("Theoretical Data", "Theoretical data loaded successfully.")

            # Redraw the plot with the new data
            update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load theoretical data:\n{e}")
    else:
        messagebox.showinfo("Load Theoretical Data", "No file selected.")
```
### File Selection and Data Processing
- **`filedialog.askopenfilename(...)`**: Opens a file selection dialog. The chosen path is stored in `file_path`.
- **`os.path.exists(file_path)`**: Confirms the path is valid.
- **`pd.read_csv(...)`**: Reads in the file as a whitespace-delimited file (instead of comma-delimited). The code expects no header in the raw data, hence `header=None`. It explicitly names the columns `["Energy", "OscillatorStrength"]`.
- **`pd.to_numeric(..., errors="coerce")`**: Ensures columns are numeric. If a non-numeric value appears, it becomes `NaN`.
- **`df.dropna(inplace=True)`**: Removes any rows that contain `NaN`, ensuring clean data.

### Wavelength Conversion

The wavelength is calculated using the formula:

$$
\lambda (\text{nm}) = \frac{1239.84}{E (\text{eV})}
$$

where **1239.84 nm·eV** is a commonly used approximate constant.

### Sorting and Finalizing the Data

- **`df.sort_values("Wavelength")`**: Sorts the data by ascending wavelength, which makes plotting more orderly.
- **`theoretical_df = df.reset_index(drop=True)`**: Stores the cleaned and prepared DataFrame in the global variable. `reset_index(drop=True)` discards the old index and gives a fresh `0,1,2,...` index.

### Updating the Plot and Error Handling

- **`update_plot()`**: Once the data is loaded, the plot is updated.
- If an error occurs (such as a read failure), a `messagebox` displays the error details.

## 3. Loading Experimental Data

```python
def load_experimental_data():
    global experimental_df
    file_path = filedialog.askopenfilename(title="Select Experimental Data (experimental.txt)")
    if file_path and os.path.exists(file_path):
        try:
            # Read experimental data (assuming two columns: Wavelength (nm) and Absorption)
            df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["Wavelength", "Absorption"])
            df["Wavelength"] = pd.to_numeric(df["Wavelength"], errors="coerce")
            df["Absorption"] = pd.to_numeric(df["Absorption"], errors="coerce")
            df.dropna(inplace=True)

            # Sort by wavelength (ascending)
            df.sort_values("Wavelength", inplace=True)

            # Save DataFrame to the global variable
            experimental_df = df.reset_index(drop=True)

            # Notify success and update the plot
            messagebox.showinfo("Experimental Data", "Experimental data loaded successfully.")
            update_plot()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load experimental data:\n{e}")
    else:
        messagebox.showinfo("Load Experimental Data", "No file selected.")
```

## 4.Generating the Theoretical Spectrum

```python
def generate_theoretical_spectrum(fwhm, grid=None):
    """
    Build a continuous theoretical absorption spectrum by broadening each discrete transition 
    (each given by a wavelength and oscillator strength) with a Gaussian of the given FWHM.
    """
    
    # If no theoretical data is loaded or it's empty, return (None, None).
    if theoretical_df is None or theoretical_df.empty:
        return None, None

    # Determine the minimum and maximum wavelengths, adding a 20 nm buffer on each side.
    wl_min = theoretical_df["Wavelength"].min() - 20
    wl_max = theoretical_df["Wavelength"].max() + 20

    # If no custom grid is provided, create a default grid of 2000 points spanning wl_min to wl_max.
    if grid is None:
        grid = np.linspace(wl_min, wl_max, 2000)

    # Convert the Full Width at Half Maximum (FWHM) to the standard deviation (sigma) of a Gaussian.
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Initialize the spectrum array as zeros, having the same shape as the grid.
    spectrum = np.zeros_like(grid)

    # Loop through each row in the theoretical data to add a Gaussian contribution for each transition.
    for _, row in theoretical_df.iterrows():
        # Extract the transition's wavelength.
        wl = row["Wavelength"]
        # Extract the transition's oscillator strength.
        strength = row["OscillatorStrength"]
        # Add the Gaussian contribution for this transition onto the spectrum.
        spectrum += strength * np.exp(-0.5 * ((grid - wl) / sigma)**2)

    # Return the wavelength grid along with the final theoretical spectrum array.
    return grid, spectrum
```
### Checking if `theoretical_df` is Loaded

- If no theoretical data is present, the function returns `(None, None)`.

### Determining the Wavelength Range

- Compute the minimum and maximum wavelength (`wl_min` and `wl_max`) from the DataFrame.
- Add an extra 20 nm padding on each side.

### Creating the Grid

- If no grid is provided, create a default grid using `np.linspace(...)` with 2000 points between `wl_min` and `wl_max`.

### Converting FWHM to Standard Deviation

The standard deviation σ is calculated from the full width at half maximum (FWHM) using the formula:

$$
\sigma = \frac{\text{FWHM}}{2 \sqrt{2 \ln 2}}
$$

This is the standard formula for a Gaussian’s FWHM.

### Initializing the Spectrum

- Create a spectrum array filled with zeros, with the same length as the grid.

### Looping Through `theoretical_df`

For each row in `theoretical_df`:
- **`wl`** = Wavelength of the transition
- **`strength`** = Oscillator strength

Add a Gaussian “peak” to the spectrum using:

$$
\text{peak} = \text{strength} \times \exp \left( -0.5 \left( \frac{\text{grid} - wl}{\sigma} \right)^2 \right)
$$

The summation of all individual Gaussian peaks forms a continuous spectrum.

## 5. Updating (Redrawing) the Plot

```python
def update_plot(event=None):
    # Get parameters from UI
    try:
        current_fwhm = float(fwhm_scale.get())
    except:
        current_fwhm = 1.0
    try:
        shift_val = float(shift_scale.get())  # shift in eV
    except:
        shift_val = 0.0

    # Clear the plot
    ax.clear()

    # Plot theoretical spectrum if available
    grid, theo_spec = generate_theoretical_spectrum(current_fwhm)
    if grid is not None and theo_spec is not None:
        ax.plot(grid, theo_spec, label="Theoretical", color="orange", lw=2)

    # Plot experimental data if available (apply shift in eV)
    if experimental_df is not None and not experimental_df.empty:
        shifted_exp = experimental_df.copy()
        # Convert experimental wavelength (nm) to energy (eV)
        shifted_exp["Energy"] = 1239.84 / shifted_exp["Wavelength"]
        
        # Apply the energy shift
        shifted_exp["ShiftedEnergy"] = shifted_exp["Energy"] + shift_val
        
        # Convert shifted energy back to wavelength (nm)
        shifted_exp["ShiftedWavelength"] = 1239.84 / shifted_exp["ShiftedEnergy"]
        
        # Plot
        ax.plot(shifted_exp["ShiftedWavelength"], shifted_exp["Absorption"],
                label="Experimental", color="blue", marker="None", ls="-")

    # Customize plot labels and appearance
    ax.set_xlabel("Wavelength (nm)", fontsize=14)
    ax.set_ylabel("Absorption / Oscillator Strength", fontsize=14)
    ax.set_title("Comparison of Theoretical and Experimental Spectra", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    # Redraw the canvas
    canvas.draw()
```


Retrieve the GUI parameters:  
**current_fwhm** from the Gaussian FWHM scale. If conversion fails, it defaults to **1.0 nm**.  
**shift_val** from the shift scale (applied in eV). If conversion fails, it defaults to **0.0**.  

**ax.clear()**: Clears the existing axes so we can draw a fresh plot.  

Generate the theoretical spectrum using **generate_theoretical_spectrum(current_fwhm)**. If valid data exist, plot it (orange line, **lw=2** sets line width).  

Plot the experimental data if it exists:  
- Convert from **Wavelength (nm) → Energy (eV)** using **1239.84 / Wavelength**.  
- Add the shift in **eV** to the energy.  
- Convert it back to shifted wavelength to plot on the same wavelength axis as the theoretical data.  
- Plot the shifted experimental data (blue line).  

Set **labels, title, legend, and grid** with some style adjustments.  

**canvas.draw()**: Redraws the **Matplotlib canvas** in the Tkinter window so the user can see the updated plot.

## 6. Saving the Theoretical Spectrum

```python
def save_spectrum():
    # Save the current theoretical spectrum on the generated grid
    grid, theo_spec = generate_theoretical_spectrum(float(fwhm_scale.get()))
    if grid is None or theo_spec is None:
        messagebox.showerror("Error", "No theoretical data available to save.")
        return
    
    # Create a DataFrame with the wavelength and the spectral values
    save_df = pd.DataFrame({"Wavelength": grid, "TheoreticalSpectrum": theo_spec})
    
    # Ask the user where to save the file
    file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             title="Save Theoretical Spectrum")
    if file_path:
        try:
            save_df.to_csv(file_path, index=False)
            messagebox.showinfo("Save Spectrum", "Spectrum saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save spectrum:\n{e}")
```
Generate the theoretical spectrum with the current FWHM from the GUI scale.  
If no theoretical data is available, display an error and stop.  
Otherwise, create a DataFrame containing the grid (wavelength array) and the theoretical spectrum values.  
Use a “Save As” dialog to let the user pick a filename. The default extension is **.csv**.  
Write out the DataFrame to a CSV file and inform the user of success (or failure).  


## 7. Building the Tkinter UI

```python
# Create main application window
root = tk.Tk()
root.title("Spectrum Matcher")

# Create a frame for the plot
plot_frame = tk.Frame(root)
plot_frame.pack(fill=tk.BOTH, expand=True)

# Create a Matplotlib figure and canvas
fig, ax = plt.subplots(figsize=(10, 5))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
```
root = tk.Tk(): The main application window.

root.title("Spectrum Matcher"): Sets the window’s title bar text.

plot_frame = tk.Frame(root): A Tkinter Frame widget that holds the Matplotlib figure.

fig, ax = plt.subplots(figsize=(10, 5)): Creates a Matplotlib figure (fig) and axes (ax).

FigureCanvasTkAgg(fig, master=plot_frame): Embeds the Matplotlib figure into the plot_frame so it’s visible in the Tkinter interface.

pack(...): Geometry manager that places the widget in the GUI.

## 8. Control Frame with Buttons and Sliders

```python
# Create a toolbar frame for controls
control_frame = tk.Frame(root)
control_frame.pack(fill=tk.X)

# Buttons for loading data
load_theo_button = tk.Button(control_frame, text="Load Theoretical Data", command=load_theoretical_data)
load_theo_button.pack(side=tk.LEFT, padx=5, pady=5)

load_exp_button = tk.Button(control_frame, text="Load Experimental Data", command=load_experimental_data)
load_exp_button.pack(side=tk.LEFT, padx=5, pady=5)

save_button = tk.Button(control_frame, text="Save Theoretical Spectrum", command=save_spectrum)
save_button.pack(side=tk.LEFT, padx=5, pady=5)

# Scale for Gaussian FWHM (nm)
fwhm_scale = tk.Scale(control_frame, from_=0.1, to=150, resolution=0.1,
                      orient=tk.HORIZONTAL, label="Gaussian FWHM (nm)",
                      command=update_plot)
fwhm_scale.set(0.1)
fwhm_scale.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

# Scale for experimental x-axis shift (eV)
shift_scale = tk.Scale(control_frame, from_=-1, to=1, resolution=0.01,
                       orient=tk.HORIZONTAL, label="Experimental Shift (eV)",
                       command=update_plot)
shift_scale.set(0)
shift_scale.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
```

control_frame: A separate frame at the bottom or top for user controls.

Buttons:

Load Theoretical Data: calls load_theoretical_data().

Load Experimental Data: calls load_experimental_data().

Save Theoretical Spectrum: calls save_spectrum().

Scales (sliders):

FWHM scale: lets the user set the Full Width at Half Maximum (in nm) for broadening the theoretical spectrum. Minimum is 0.1 nm, maximum is 150 nm, with a 0.1 nm step. 

Setting it calls update_plot() automatically so the plot refreshes with the new value.

Shift scale: shifts the experimental data along the energy axis in eV. Ranges from -1 eV to +1 eV with 0.01 eV increments. Also calls update_plot() on movement.

## 9. Initial Plot and Mainloop

```python
# Initial call to draw empty plot
update_plot()

root.mainloop()
```

update_plot(): Called once at startup to display an empty or default plot (if no data is loaded yet).

root.mainloop(): Starts the Tkinter event loop, which keeps the GUI running and responsive. This line must be at the end of the script or the GUI will close immediately.

## Summary
The user interface is built with Tkinter, placing a Matplotlib plot in one section and controls (buttons, sliders) in another.

Data loading is done via the “Load” buttons:

Theoretical data is assumed to be (Energy, OscillatorStrength).

Experimental data is assumed to be (Wavelength, Absorption).

Theoretical spectrum is generated by applying Gaussian broadening to each discrete transition, controlled by an FWHM slider.

Experimental data can be shifted in the energy domain (in eV) before converting back to wavelength for comparison.

Plot updates happen every time data is loaded or a slider changes via update_plot().

Saving the theoretical spectrum writes the broadening result to a user-selected CSV file.

This design allows direct comparison between a calculated (theoretical) spectrum and an experimental (measured) spectrum, with the ability to adjust broadening (FWHM) and an energy shift for alignment.






