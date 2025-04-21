# -*- coding: utf-8 -*-
"""
PyQt5 application to visualize quantum mechanical standing waves:
1. Particle in a 1D infinite potential well.
2. Radial wavefunctions and probability densities for the Hydrogen atom.

Demonstrates concepts like quantization, wavefunctions (Ψ),
probability density (Ψ² or r²R²), and the role of quantum numbers.
"""

import sys
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QSpinBox, QSizePolicy  # Removed unused: QSlider, QGridLayout, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Matplotlib imports for embedding plots
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
# Removed unused: import matplotlib.pyplot as plt

# Scipy imports for special functions and constants
from scipy.special import genlaguerre, factorial
from scipy.constants import physical_constants

# Type checking imports
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from matplotlib.axes import Axes


# --- Physics Constants ---
# Using scipy's constants for precision and clear sourcing
try:
    h_bar: float = physical_constants['Planck constant over 2 pi'][0]  # J*s
    m_e: float = physical_constants['electron mass'][0]              # kg
    a0: float = physical_constants['Bohr radius'][0]                # meters (bohr)
    e_charge: float = physical_constants['elementary charge'][0]    # Coulombs
except KeyError as e:
    print(f"Error retrieving physical constant: {e}. Check scipy installation.", file=sys.stderr)
    sys.exit(1)


# --- Helper Functions ---

def particle_in_box_psi(n: int, L: float, x: 'NDArray[np.float64]') -> 'NDArray[np.float64]':
    """
    Calculate the normalized wavefunction for a particle in a 1D infinite potential well.

    Args:
        n: Principal quantum number (n = 1, 2, 3, ...).
        L: Width of the box (in meters).
        x: Numpy array of position coordinates (in meters).

    Returns:
        Numpy array containing the wavefunction values Psi(x). Returns zeros if n <= 0 or L <= 0.

    Physics:
        Psi_n(x) = sqrt(2/L) * sin(n * pi * x / L) for 0 < x < L
        Psi_n(x) = 0 otherwise
    """
    if n <= 0 or L <= 0:
        # Invalid quantum number or box size
        return np.zeros_like(x)

    norm_factor: float = np.sqrt(2.0 / L)
    psi: 'NDArray[np.float64]' = norm_factor * np.sin(n * np.pi * x / L)

    # Ensure psi is exactly zero outside and at the boundaries
    psi[x <= 0] = 0.0
    psi[x >= L] = 0.0
    return psi

def hydrogen_radial_wavefunction(n: int, l: int, r: 'NDArray[np.float64]') -> 'NDArray[np.float64]':
    """
    Calculate the normalized radial wavefunction R_nl(r) for the Hydrogen atom.

    Args:
        n: Principal quantum number (n = 1, 2, 3, ...).
        l: Angular momentum (azimuthal) quantum number (l = 0, 1, ..., n-1).
        r: Numpy array of radial distances from the nucleus (in meters).

    Returns:
        Numpy array containing the radial wavefunction values R_nl(r) (units: m^(-3/2)).
        Returns zeros if quantum numbers are invalid (n <= 0 or l >= n).

    Physics:
        R_nl(r) = N * exp(-rho/2) * rho^l * L_{n-l-1}^{2l+1}(rho)
        where rho = 2*r / (n*a0), N is a normalization constant, and
        L_{m}^{alpha} is the generalized Laguerre polynomial.
    """
    if l >= n or n <= 0:
        # Invalid quantum numbers
        return np.zeros_like(r)

    # Avoid division by zero or log(0) issues if n or a0 were ever zero (though unlikely here)
    if n == 0 or a0 == 0:
         return np.zeros_like(r)

    # Reduced radius variable rho (dimensionless)
    rho: 'NDArray[np.float64]' = 2.0 * r / (n * a0)

    # Normalization constant calculation (units: m^(-3/2))
    # Factorials can grow large, but n is typically small here. Using scipy's factorial.
    try:
        # Use float for calculations involving factorials to avoid potential overflow with large ints
        fact_nl = float(factorial(n - l - 1))
        fact_n_plus_l = float(factorial(n + l))

        # Check for potentially problematic factorial results (e.g., inf)
        if not np.isfinite(fact_nl) or not np.isfinite(fact_n_plus_l) or fact_n_plus_l == 0:
             print(f"Warning: Factorial calculation issue for n={n}, l={l}", file=sys.stderr)
             return np.zeros_like(r) # Return zeros if factorial is problematic

        norm_factor_sq = ((2.0 / (n * a0))**3) * (fact_nl / (2.0 * n * (fact_n_plus_l**3)))

        # Ensure normalization factor is non-negative before sqrt
        if norm_factor_sq < 0:
            print(f"Warning: Negative value encountered in normalization calculation for n={n}, l={l}", file=sys.stderr)
            return np.zeros_like(r) # Return zeros if normalization is problematic

        norm: float = np.sqrt(norm_factor_sq)

    except ValueError: # Handles factorial of negative numbers
        print(f"Warning: Invalid arguments for factorial (n={n}, l={l})", file=sys.stderr)
        return np.zeros_like(r)


    # Associated Laguerre polynomial L_{n-l-1}^{2l+1}(rho)
    # Note: scipy's genlaguerre(m, alpha) corresponds to L_m^alpha
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)
    laguerre_values: 'NDArray[np.float64]' = laguerre_poly(rho)

    # Radial wavefunction R_nl(r)
    # Handle potential 0^0 case for r=0, l=0 -> rho=0, l=0. rho**l should be 1.
    # numpy handles 0**0 as 1, which is correct here.
    # exp(-rho/2) handles large rho resulting in small values.
    R_nl: 'NDArray[np.float64]' = norm * np.exp(-rho / 2.0) * (rho**l) * laguerre_values

    # Ensure R_nl is zero at r=0 for l>0, as expected physically.
    # Handles cases where r array might start exactly at 0.
    if l > 0:
        R_nl[r < 1e-15] = 0.0 # Use small tolerance for float comparison

    return R_nl

# --- Matplotlib Canvas Widget ---

class MplCanvas(FigureCanvas):
    """
    A custom Matplotlib canvas widget integrated into PyQt5.

    Manages a Matplotlib Figure with two subplots stacked vertically.
    Provides methods for clearing and redrawing the plots.
    """
    def __init__(self, parent: QWidget = None, width: int = 5, height: int = 4, dpi: int = 100):
        """
        Initialize the Matplotlib Figure and Axes.

        Args:
            parent: The parent Qt widget.
            width: Figure width in inches.
            height: Figure height in inches.
            dpi: Dots per inch resolution.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        # Add subplots and store references to the axes
        self.axes_upper: 'Axes' = self.fig.add_subplot(2, 1, 1) # 2 rows, 1 col, first plot
        self.axes_lower: 'Axes' = self.fig.add_subplot(2, 1, 2) # 2 rows, 1 col, second plot
        super().__init__(self.fig)
        self.setParent(parent)

        # Set Qt size policies for layout management
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)

    def clear_plots(self) -> None:
        """Clears both subplots (axes_upper and axes_lower)."""
        self.axes_upper.clear()
        self.axes_lower.clear()

    def redraw(self) -> None:
        """Redraws the figure efficiently using draw_idle."""
        self.fig.canvas.draw_idle() # Use draw_idle for better responsiveness


# --- Main Application Window ---

class StandingWaveSim(QWidget):
    """
    Main application window for the Standing Wave Simulator.

    Contains tabs for visualizing:
    1. Particle in a 1D Box wavefunctions and probability densities.
    2. Hydrogen atom radial wavefunctions and probability densities.
    Includes controls to change quantum numbers (n, l).
    """
    def __init__(self):
        super().__init__()
        # --- Simulation Parameters ---
        self.box_length_L: float = 1.0e-9    # Default: 1 nm box width (meters)
        self.radial_max_r: float = 15 * a0   # Max radius to plot for H atom (meters)
        self.num_points: int = 400           # Number of points for generating plot data

        # --- Initialize UI ---
        self.initUI()

        # --- Initial Plot Updates ---
        self.update_1d_plot()
        self.update_radial_plot()

    def initUI(self) -> None:
        """Initialize the user interface elements and layout."""
        self.setWindowTitle("Quantum Standing Waves & Atomic Orbitals Simulator")
        # Increased height slightly for better visibility of explanation
        self.setGeometry(100, 100, 950, 750)

        main_layout = QVBoxLayout(self)

        # --- Title ---
        title_label = QLabel("Quantum Standing Waves & Atomic Orbitals")
        title_font = QFont('Arial', 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # --- Explanation Box ---
        explanation_label = QLabel(
            "<b>Theme:</b> How electrons form orbitals in atoms through standing waves.<br>"
            "<b>Description:</b> This simulation visualizes standing wave patterns (wavefunctions) which are fundamental to quantum mechanics. "
            "Electrons confined by potential wells (like the Coulomb potential in atoms) can only exist in specific quantized energy states, each corresponding to an allowed standing wave. "
            "The square of the wavefunction (|Ψ|²) gives the probability density of finding the electron."
             "<br><b>Physics Covered:</b> Bound states, quantization, wavefunctions (Ψ), probability density (|Ψ|² or r²R² for radial part), atomic structure (radial component)."
        )
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet(
            "padding: 10px; "
            "background-color: #e8f4f8; " # Light blue background
            "border: 1px solid #b0dceb; "  # Slightly darker blue border
            "border-radius: 5px; "
            "margin-bottom: 10px;"
        )
        # Set size policy to ensure it takes needed vertical space and expands horizontally
        explanation_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        main_layout.addWidget(explanation_label) # Add to main layout

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        # Tab widget should expand to fill available space
        self.tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.tab_widget)

        # == Tab 1: 1D Infinite Potential Well ==
        self.tab1d = QWidget()
        tab1d_layout = QHBoxLayout(self.tab1d) # Horizontal: Controls | Plot
        self.tab_widget.addTab(self.tab1d, "1D Potential Well (Box)")

        # -- Controls for 1D Box --
        controls_1d_widget = QWidget() # Use a widget container for controls layout
        controls_1d_layout = QVBoxLayout(controls_1d_widget)
        controls_1d_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Align controls to the top

        label_n1d = QLabel("Quantum Number (n):")
        self.spinbox_n1d = QSpinBox()
        self.spinbox_n1d.setMinimum(1)
        self.spinbox_n1d.setMaximum(10) # Practical upper limit for visualization
        self.spinbox_n1d.setValue(1)
        self.spinbox_n1d.valueChanged.connect(self.update_1d_plot)
        self.spinbox_n1d.setToolTip("Select the principal quantum number n (1, 2, ...)")

        controls_1d_layout.addWidget(label_n1d)
        controls_1d_layout.addWidget(self.spinbox_n1d)
        controls_1d_layout.addStretch(1) # Pushes controls up

        # Set fixed width for control panel for better layout stability
        controls_1d_widget.setFixedWidth(150)

        # -- Plot area for 1D Box --
        self.canvas_1d = MplCanvas(self, width=7, height=6) # Adjusted size slightly
        tab1d_layout.addWidget(controls_1d_widget) # Add controls widget
        tab1d_layout.addWidget(self.canvas_1d, 1) # Add canvas, stretch factor 1

        # == Tab 2: Radial Wavefunctions (Hydrogen Atom) ==
        self.tab_radial = QWidget()
        tab_radial_layout = QHBoxLayout(self.tab_radial) # Horizontal: Controls | Plot
        self.tab_widget.addTab(self.tab_radial, "Atomic Radial Functions (Hydrogen)")

        # -- Controls for Radial Functions --
        controls_rad_widget = QWidget() # Container widget
        controls_rad_layout = QVBoxLayout(controls_rad_widget)
        controls_rad_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label_n_rad = QLabel("Principal Quantum Number (n):")
        self.spinbox_n_rad = QSpinBox()
        self.spinbox_n_rad.setMinimum(1)
        self.spinbox_n_rad.setMaximum(7) # Practical limit for stable visualization
        self.spinbox_n_rad.setValue(1)
        self.spinbox_n_rad.setToolTip("Select the principal quantum number n (1, 2, ...)")

        label_l_rad = QLabel("Angular Momentum QN (l):")
        self.spinbox_l_rad = QSpinBox()
        self.spinbox_l_rad.setMinimum(0)
        # Max l is n-1, will be updated dynamically by n_rad_changed
        self.spinbox_l_rad.setMaximum(self.spinbox_n_rad.value() - 1)
        self.spinbox_l_rad.setValue(0)
        self.spinbox_l_rad.setToolTip("Select angular momentum l (0 <= l < n)")


        # Connect signals AFTER both spinboxes are created
        self.spinbox_n_rad.valueChanged.connect(self.n_rad_changed)
        # l changes always trigger plot update
        self.spinbox_l_rad.valueChanged.connect(self.update_radial_plot)

        controls_rad_layout.addWidget(label_n_rad)
        controls_rad_layout.addWidget(self.spinbox_n_rad)
        controls_rad_layout.addWidget(label_l_rad)
        controls_rad_layout.addWidget(self.spinbox_l_rad)
        controls_rad_layout.addStretch(1) # Pushes controls up

        # Set fixed width for control panel
        controls_rad_widget.setFixedWidth(150)

        # -- Plot area for Radial Functions --
        self.canvas_radial = MplCanvas(self, width=7, height=6)
        tab_radial_layout.addWidget(controls_rad_widget) # Add controls widget
        tab_radial_layout.addWidget(self.canvas_radial, 1) # Add canvas, stretch factor 1


        # self.setLayout(main_layout) # Already set during QVBoxLayout creation
        self.show()

    # --- Update Functions ---

    def n_rad_changed(self, n_value: int) -> None:
        """
        Handles changes in the principal quantum number 'n' for Hydrogen radial plots.
        Updates the maximum allowed value for 'l' (l = 0, 1, ..., n-1) and
        triggers a plot update if 'l' remains valid, or resets 'l' if needed.
        """
        current_l = self.spinbox_l_rad.value()
        # Ensure max_l is at least 0 (for n=1 case)
        max_l = max(0, n_value - 1)
        self.spinbox_l_rad.setMaximum(max_l)

        # If the current l is now invalid (greater than max allowed), reset it.
        # Setting the value will trigger spinbox_l_rad.valueChanged, which calls update_radial_plot.
        if current_l > max_l:
            self.spinbox_l_rad.setValue(max_l)
            # No need to call update_radial_plot here, signal handles it.
        else:
            # If l is still valid, directly trigger the plot update.
            self.update_radial_plot()


    def update_1d_plot(self) -> None:
        """Updates the 1D Particle-in-a-Box plots (Psi and Psi^2)."""
        n = self.spinbox_n1d.value()
        L = self.box_length_L
        # Generate x values within the box
        x = np.linspace(0, L, self.num_points)

        psi = particle_in_box_psi(n, L, x)
        prob_density = psi**2

        # --- Plotting ---
        self.canvas_1d.clear_plots()
        ax_psi = self.canvas_1d.axes_upper
        ax_prob = self.canvas_1d.axes_lower

        # Plot Wavefunction (Psi)
        ax_psi.plot(x * 1e9, psi, label=f'Ψ(n={n})', color='blue')
        ax_psi.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax_psi.set_xlabel("Position x (nm)")
        ax_psi.set_ylabel("Wavefunction Ψ(x)")
        ax_psi.set_title(f"1D Box Wavefunction (n={n}, L={L*1e9:.1f} nm)")
        ax_psi.grid(True, linestyle=':', alpha=0.6)
        ax_psi.legend(loc='upper right')
        # Set Y limits symmetrically based on max amplitude, add padding
        max_psi_abs = np.max(np.abs(psi)) if psi.size > 0 and np.any(psi) else 1.0
        ax_psi.set_ylim(-max_psi_abs * 1.2, max_psi_abs * 1.2)
        ax_psi.set_xlim(0, L * 1e9) # Ensure x-limits match box size

        # Plot Probability Density (|Psi|^2)
        ax_prob.plot(x * 1e9, prob_density, label=f'|Ψ(n={n})|²', color='red')
        ax_prob.fill_between(x * 1e9, 0, prob_density, color='red', alpha=0.3)
        ax_prob.set_xlabel("Position x (nm)")
        ax_prob.set_ylabel("Probability Density |Ψ(x)|²")
        # ax_prob.set_title(f"1D Box Probability Density (n={n})") # Title redundant with lower axis
        ax_prob.grid(True, linestyle=':', alpha=0.6)
        ax_prob.legend(loc='upper right')
        ax_prob.set_ylim(bottom=0) # Probability density cannot be negative
        ax_prob.set_xlim(0, L * 1e9) # Ensure x-limits match box size

        self.canvas_1d.redraw()


    def update_radial_plot(self) -> None:
        """Updates the Hydrogen Atom Radial plots (R_nl and P(r)=r^2*R_nl^2)."""
        n = self.spinbox_n_rad.value()
        l = self.spinbox_l_rad.value()

        # Safeguard: Ensure l is valid (although n_rad_changed should handle this)
        if l >= n:
             l = max(0, n - 1)
             # Update spinbox without emitting signal again if possible, or be aware of recursion
             self.spinbox_l_rad.blockSignals(True)
             self.spinbox_l_rad.setValue(l)
             self.spinbox_l_rad.blockSignals(False)
             # If l was invalid, we probably don't want to plot garbage anyway,
             # maybe return early or plot zeros. Here we correct l and continue.
             print(f"Warning: Corrected invalid l={self.spinbox_l_rad.value()+1} to l={l} for n={n}.", file=sys.stderr)


        # Define radius range, starting slightly > 0 to avoid potential issues at r=0
        # r units: meters
        r = np.linspace(1e-15, self.radial_max_r, self.num_points)
        # r in units of Bohr radius (a0) for plotting
        r_bohr = r / a0

        R_nl = hydrogen_radial_wavefunction(n, l, r) # Units: m^(-3/2)
        # Radial probability density P(r) = r^2 * |R_nl(r)|^2
        # Units: (m^2) * (m^-3) = m^-1 (Probability per unit radius)
        radial_prob_density = (r**2) * (R_nl**2)

        # --- Plotting ---
        self.canvas_radial.clear_plots()
        ax_R = self.canvas_radial.axes_upper
        ax_P = self.canvas_radial.axes_lower

        # Plot Radial Wavefunction R_nl(r)
        ax_R.plot(r_bohr, R_nl, label=f'R(n={n}, l={l})', color='purple')
        ax_R.axhline(0, color='gray', linestyle='--', linewidth=0.7)
        ax_R.set_xlabel("Radius r (units of a₀)")
        ax_R.set_ylabel(f"Radial Wavefunction R(r) [m⁻³/²]")
        ax_R.set_title(f"Hydrogen Radial Wavefunction (n={n}, l={l})")
        ax_R.grid(True, linestyle=':', alpha=0.6)
        ax_R.legend(loc='upper right')
        # Dynamically adjust y-limits based on data range, adding padding
        min_R, max_R = (np.min(R_nl), np.max(R_nl)) if R_nl.size > 0 and np.any(np.isfinite(R_nl)) else (-1, 1)
        if np.isclose(min_R, max_R): # Handle case where R_nl is constant (e.g., zero)
             padding_R = 0.1
        else:
             padding_R = (max_R - min_R) * 0.1
        ax_R.set_ylim(min_R - padding_R, max_R + padding_R)
        ax_R.set_xlim(0, self.radial_max_r / a0) # Set x limit based on max radius

        # Plot Radial Probability Density P(r) = r^2 * R_nl^2
        # Often, P(r) is plotted against r/a0, so P(r) needs scaling if we want area under
        # the P(r) vs r/a0 curve to be 1. P(r)dr = P(r) * a0 * d(r/a0).
        # So, plot a0*P(r) vs r/a0 to get a dimensionless probability density per a0.
        P_scaled = a0 * radial_prob_density # Dimensionless probability density per a0 unit
        ax_P.plot(r_bohr, P_scaled, label=f'a₀·r²|R(n={n}, l={l})|²', color='orange')
        ax_P.fill_between(r_bohr, 0, P_scaled, color='orange', alpha=0.3)
        ax_P.set_xlabel("Radius r (units of a₀)")
        ax_P.set_ylabel("Radial Prob. Density P(r)·a₀ [unitless]")
        # ax_P.set_title(f"Hydrogen Radial Probability Density (n={n}, l={l})") # Title redundant
        ax_P.grid(True, linestyle=':', alpha=0.6)
        ax_P.legend(loc='upper right')
        ax_P.set_ylim(bottom=0) # Probability density >= 0
        # Optional: Set dynamic upper y-limit for P(r) plot if needed
        max_P_scaled = np.max(P_scaled) if P_scaled.size > 0 and np.any(np.isfinite(P_scaled)) else 0.1
        ax_P.set_ylim(bottom=0, top=max_P_scaled * 1.15) # Add 15% padding
        ax_P.set_xlim(0, self.radial_max_r / a0) # Set x limit

        self.canvas_radial.redraw()


# --- Run the Application ---
if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the main window
    ex = StandingWaveSim()

    # Run the main Qt loop
    sys.exit(app.exec_())
