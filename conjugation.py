import sys
import math
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QGraphicsView, QGraphicsScene, QGraphicsTextItem,
    QFrame, QGridLayout
)
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPen, QColor, QFont, QPainter

# --- Constants ---
H_PLANCK = 6.626e-34  # Planck's constant (J*s)
M_ELECTRON = 9.109e-31 # Electron mass (kg)
C_LIGHT = 2.998e8     # Speed of light (m/s)
E_CHARGE = 1.602e-19  # Elementary charge (C) for eV conversion

# Effective parameters for the 1D box model (can be adjusted)
EFFECTIVE_LENGTH_PER_UNIT = 2.8e-10 # meters (e.g., 2 * 1.4 Å)
BOX_END_EXTENSION = 1.4e-10        # Add length at ends (like half bond length)

# --- Helper Functions ---

def calculate_energy_levels(n_quantum, L_box):
    """Calculates energy level in Joules for a given quantum number n and box length L."""
    if L_box <= 0:
        return 0
    # E_n = (n^2 * h^2) / (8 * m * L^2)
    energy_J = (n_quantum**2 * H_PLANCK**2) / (8 * M_ELECTRON * L_box**2)
    return energy_J

def wavelength_to_color(wavelength_nm):
    """Maps wavelength (nm) to absorbed color name, RGB, and perceived color name."""
    if wavelength_nm < 380:
        return "UV", QColor(128, 0, 128), "Colorless / Pale Yellow (Perceived)" # Often perceived complement is pale yellow
    elif 380 <= wavelength_nm < 450:
        return "Violet", QColor(128, 0, 128), "Yellow-Green (Perceived)"
    elif 450 <= wavelength_nm < 495:
        return "Blue", QColor(0, 0, 255), "Yellow (Perceived)"
    elif 495 <= wavelength_nm < 570:
        return "Green", QColor(0, 255, 0), "Red/Purple (Perceived)"
    elif 570 <= wavelength_nm < 590:
        return "Yellow", QColor(255, 255, 0), "Blue (Perceived)"
    elif 590 <= wavelength_nm < 620:
        return "Orange", QColor(255, 165, 0), "Blue-Green (Perceived)"
    elif 620 <= wavelength_nm <= 750:
        return "Red", QColor(255, 0, 0), "Green (Perceived)"
    else:
        return "Infrared (IR)", QColor(100, 100, 100), "Colorless (Perceived)"

# --- Main Application Window ---

class ConjugatedSystemSim(QWidget):
    def __init__(self):
        super().__init__()
        self.n_units = 1  # Initial number of conjugated units (double bonds)
        self.initUI()
        self.update_simulation() # Initial calculation

    def initUI(self):
        self.setWindowTitle("Conjugated System & Color Perception Simulator")
        self.setGeometry(100, 100, 850, 700) # Increased height slightly for equation

        # --- Layouts ---
        main_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout() # For controls and explanation
        bottom_layout = QHBoxLayout() # For visualization and results

        control_panel = QVBoxLayout()
        results_panel = QVBoxLayout()
        vis_panel = QVBoxLayout()

        # --- Styling ---
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0; /* Light gray background */
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QLabel#TitleLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #2c3e50; /* Dark blue-gray */
                padding-bottom: 5px; /* Reduced padding */
            }
            /* Style for the equation */
            QLabel#EquationLabel {
                font-size: 15pt; /* Larger font for equation */
                font-weight: bold;
                color: #16a085; /* Teal color for emphasis */
                padding-top: 5px;
                padding-bottom: 15px; /* Space below equation */
                /* Use a font that supports mathematical symbols well if needed */
                /* font-family: "DejaVu Sans", Arial, sans-serif; */
            }
            QLabel#ExplanationLabel {
                padding: 10px;
                background-color: #eaf2f8; /* Light blue background */
                border: 1px solid #aed6f1;
                border-radius: 5px;
                color: #2c3e50; /* Darker text color for better contrast */
                /* font-size: 9.5pt; Slightly smaller if needed */
            }
            QLabel#HeaderLabel {
                font-size: 11pt;
                font-weight: bold;
                color: #34495e; /* Slightly lighter blue-gray */
                margin-top: 10px;
                margin-bottom: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3498db; /* Nice blue */
                border: 1px solid #2980b9;
                width: 18px;
                margin: -5px 0; /* Adjust vertical position */
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #3498db;
                border: 1px solid #bbb;
                height: 8px;
                border-radius: 4px;
            }
            QGraphicsView {
                border: 1px solid #bdc3c7; /* Silver border */
                background-color: white;
            }
            QLabel#ColorBoxLabel {
                border: 1px solid #7f8c8d; /* Gray border */
                min-height: 50px;
                max-height: 50px;
                min-width: 150px;
                font-weight: bold;
                color: black; /* Default text color */
                alignment: AlignCenter;
                padding: 5px;
            }
            QLabel#ResultValueLabel {
                font-weight: bold;
                color: #c0392b; /* Reddish color for results */
                font-size: 11pt;
            }
            QFrame#Divider {
                background-color: #bdc3c7; /* Silver */
                min-height: 2px;
                max-height: 2px;
            }
        """)

        # --- Title ---
        title_label = QLabel("Conjugated System & Color Perception")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)

        # --- Equation Display ---
        # Using HTML for subscripts and superscripts
        equation_text = "E<sub>n</sub> = (n<sup>2</sup>h<sup>2</sup>) / (8mL<sup>2</sup>)"
        # Alternative using Unicode: "E\u2099 = (n\u00B2h\u00B2) / (8mL\u00B2)"
        self.equation_label = QLabel(equation_text)
        self.equation_label.setObjectName("EquationLabel")
        self.equation_label.setAlignment(Qt.AlignCenter)
        self.equation_label.setToolTip("Particle-in-a-1D-Box Energy Levels\n"
                                      "E = Energy\nn = Quantum Number (integer)\n"
                                      "h = Planck's Constant\nm = Electron Mass\n"
                                      "L = Length of the Box (Conjugated System)")


        # --- Explanation / Theme ---
        explanation_text = QLabel(
            "<b>Theme:</b> Why are polyphenols in fruits colorful?<br>"
            "<b>Model:</b> Electrons in conjugated π-systems behave like particles in a 1D box. "
            "Longer conjugation (more double bonds, N) means a longer 'box' (L).<br>"
            "This simulation shows how changing L affects electron energy levels (E<sub>n</sub>), "
            "the energy gap (ΔE) for HOMO → LUMO transitions, the wavelength (λ) absorbed, "
            "and the resulting color we perceive."
        )
        explanation_text.setObjectName("ExplanationLabel") # Assign object name for styling
        explanation_text.setWordWrap(True)
        explanation_text.setAlignment(Qt.AlignTop)
        # Styling moved to main stylesheet using #ExplanationLabel

        # --- Controls ---
        control_panel.addWidget(QLabel("Adjust Conjugated Chain Length (N):", objectName="HeaderLabel"))
        control_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)  # Min N=1 (e.g., ethene)
        self.slider.setMaximum(15) # Max N=15 (adjust as needed)
        self.slider.setValue(self.n_units)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.slider_changed)

        self.slider_label = QLabel(f"N = {self.n_units}")
        self.slider_label.setMinimumWidth(80) # Ensure space for label

        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.slider_label)
        control_panel.addLayout(control_layout)
        control_panel.addStretch(1) # Push controls to top

        # --- Energy Level Visualization ---
        vis_panel.addWidget(QLabel("Energy Levels (HOMO/LUMO)", objectName="HeaderLabel"))
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing) # Smooth lines
        self.view.setMinimumHeight(300)
        vis_panel.addWidget(self.view)

        # --- Results Panel ---
        results_panel.addWidget(QLabel("Calculation Results:", objectName="HeaderLabel"))
        results_grid = QGridLayout()
        results_grid.setSpacing(10)

        results_grid.addWidget(QLabel("Box Length (L):"), 0, 0)
        self.length_label = QLabel("N/A")
        self.length_label.setObjectName("ResultValueLabel")
        results_grid.addWidget(self.length_label, 0, 1)

        results_grid.addWidget(QLabel("HOMO Level (n):"), 1, 0)
        self.homo_n_label = QLabel("N/A")
        self.homo_n_label.setObjectName("ResultValueLabel")
        results_grid.addWidget(self.homo_n_label, 1, 1)

        results_grid.addWidget(QLabel("LUMO Level (n):"), 2, 0)
        self.lumo_n_label = QLabel("N/A")
        self.lumo_n_label.setObjectName("ResultValueLabel")
        results_grid.addWidget(self.lumo_n_label, 2, 1)

        results_grid.addWidget(QLabel("Energy Gap (ΔE):"), 3, 0)
        self.energy_gap_label = QLabel("N/A")
        self.energy_gap_label.setObjectName("ResultValueLabel")
        results_grid.addWidget(self.energy_gap_label, 3, 1)

        results_grid.addWidget(QLabel("Absorbed λ:"), 4, 0)
        self.wavelength_label = QLabel("N/A")
        self.wavelength_label.setObjectName("ResultValueLabel")
        results_grid.addWidget(self.wavelength_label, 4, 1)

        results_panel.addLayout(results_grid)
        results_panel.addWidget(QFrame(objectName="Divider")) # Visual Separator

        # --- Color Visualization ---
        results_panel.addWidget(QLabel("Color Absorption & Perception:", objectName="HeaderLabel"))
        color_layout = QHBoxLayout()

        self.absorbed_color_label = QLabel("Absorbed\nColor")
        self.absorbed_color_label.setObjectName("ColorBoxLabel")
        self.absorbed_color_label.setAlignment(Qt.AlignCenter)

        self.perceived_color_label = QLabel("Perceived\nColor")
        self.perceived_color_label.setObjectName("ColorBoxLabel")
        self.perceived_color_label.setAlignment(Qt.AlignCenter)

        color_layout.addWidget(self.absorbed_color_label)
        color_layout.addWidget(self.perceived_color_label)
        results_panel.addLayout(color_layout)
        results_panel.addStretch(1) # Push results to top

        # --- Assemble Layouts ---
        top_layout.addLayout(control_panel, stretch=1)
        top_layout.addWidget(explanation_text, stretch=2) # Give more space to explanation

        bottom_layout.addLayout(vis_panel, stretch=2) # More space for energy diagram
        bottom_layout.addLayout(results_panel, stretch=1)

        main_layout.addWidget(title_label)
        main_layout.addWidget(self.equation_label) # Add equation label here
        main_layout.addLayout(top_layout)
        main_layout.addWidget(QFrame(objectName="Divider")) # Visual Separator
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.show()

    def slider_changed(self, value):
        self.n_units = value
        self.slider_label.setText(f"N = {self.n_units}")
        self.update_simulation()

    def update_simulation(self):
        # 1. Calculate Box Length (L)
        L_box = self.n_units * EFFECTIVE_LENGTH_PER_UNIT + BOX_END_EXTENSION
        self.length_label.setText(f"{L_box*1e9:.2f} nm") # Display in nm

        # 2. Determine HOMO and LUMO quantum numbers (n)
        n_homo = self.n_units
        n_lumo = self.n_units + 1
        self.homo_n_label.setText(f"{n_homo}")
        self.lumo_n_label.setText(f"{n_lumo}")

        # 3. Calculate Energies
        E_homo_J = calculate_energy_levels(n_homo, L_box)
        E_lumo_J = calculate_energy_levels(n_lumo, L_box)

        # Handle case where L=0 or N=0
        if E_lumo_J == 0 or E_homo_J == E_lumo_J:
             delta_E_J = 0
             wavelength_m = float('inf')
        else:
            delta_E_J = E_lumo_J - E_homo_J
            if delta_E_J > 0:
                wavelength_m = (H_PLANCK * C_LIGHT) / delta_E_J
            else:
                wavelength_m = float('inf')

        wavelength_nm = wavelength_m * 1e9 # Convert to nm

        # 5. Update Result Labels
        delta_E_eV = delta_E_J / E_CHARGE
        self.energy_gap_label.setText(f"{delta_E_eV:.2f} eV")

        if wavelength_nm == float('inf') or wavelength_nm < 0 : # Added check for negative
             self.wavelength_label.setText("N/A (IR/UV/Large λ)")
             absorb_name, absorb_color, perceive_name = "N/A", QColor("lightgrey"), "N/A"
        else:
             self.wavelength_label.setText(f"{wavelength_nm:.1f} nm")
             # 6. Determine Colors
             absorb_name, absorb_color, perceive_name = wavelength_to_color(wavelength_nm)


        # Style the absorbed color box
        absorb_style = f"background-color: {absorb_color.name()};"
        brightness = (absorb_color.red() * 299 + absorb_color.green() * 587 + absorb_color.blue() * 114) / 1000
        text_color = "black" if brightness > 128 else "white"
        absorb_style += f" color: {text_color};"
        self.absorbed_color_label.setStyleSheet("QLabel#ColorBoxLabel {" + absorb_style + "}")
        self.absorbed_color_label.setText(f"Absorbed:\n{absorb_name}")

        # Style the perceived color box
        self.perceived_color_label.setStyleSheet("QLabel#ColorBoxLabel { background-color: #ffffff; color: black; }") # Reset style
        self.perceived_color_label.setText(f"{perceive_name}")


        # 7. Update Energy Level Diagram
        self.draw_energy_levels(n_homo, n_lumo, E_homo_J, E_lumo_J)

    def draw_energy_levels(self, n_homo, n_lumo, E_homo_J, E_lumo_J):
        self.scene.clear()
        view_rect = self.view.rect()
        # Adjust padding slightly if needed
        scene_width = view_rect.width() - 30 # More horizontal padding
        scene_height = view_rect.height() - 50 # More vertical padding for labels

        if view_rect.width() <= 30 or view_rect.height() <= 50: # Avoid drawing if view is too small
            return

        # --- Energy Scaling ---
        # Ensure LUMO energy is always treated as higher than HOMO for scaling, even if calc is weird
        E_high = max(E_lumo_J, E_homo_J)
        E_low = min(E_lumo_J, E_homo_J)
        if E_high <= 0 : # If both are zero or negative, create a default small positive range
             E_high = calculate_energy_levels(2, BOX_END_EXTENSION) # Estimate E2 for minimal box
             E_low = 0
        elif E_low < 0: # If HOMO is negative (unphysical but possible in model), start scale at E_low
             E_low = E_low * 1.1 # Go slightly lower
        else:
             E_low = E_low * 0.9 # Start scale slightly below HOMO

        # Ensure max energy is significantly above E_high
        E_top = E_high * 1.2 if E_high > 0 else E_low + H_PLANCK**2/(8*M_ELECTRON*BOX_END_EXTENSION**2) # Add arbitrary energy if E_high is 0

        # Prevent zero or negative energy range
        energy_range = E_top - E_low
        if energy_range <= 1e-22: # Use a small positive value to avoid division by zero
            energy_range = 1e-22

        # Prevent extreme scaling if E_low is very close to E_top
        if abs(E_homo_J - E_lumo_J) / energy_range < 0.01 and E_homo_J != E_lumo_J:
             energy_range *= 5 # Artificially increase range if levels are too close

        # --- Coordinate Mapping ---
        # Map Energy (Joules) to Y-coordinate (Scene coords: 0,0 top-left)
        def energy_to_y(E_J):
            # Clamp energy values to the defined scale to prevent drawing outside bounds
            clamped_E = max(E_low, min(E_J, E_top))
            # Higher energy -> lower Y
            y_pos = scene_height - ((clamped_E - E_low) / energy_range * scene_height)
            return max(0, min(scene_height, y_pos)) # Ensure y is within 0 to scene_height

        # --- Drawing Parameters ---
        line_pen = QPen(QColor("#34495e"), 2) # Dark blue-gray lines
        arrow_pen = QPen(QColor("#e74c3c"), 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin) # Red arrow, slightly thicker
        text_font = QFont("Segoe UI", 9) # Slightly larger font
        label_color = QColor("#2c3e50")
        level_line_x_start = 20
        level_line_x_end = scene_width - 20

        # --- Draw HOMO Level ---
        y_homo = energy_to_y(E_homo_J)
        self.scene.addLine(level_line_x_start, y_homo, level_line_x_end, y_homo, line_pen)
        homo_text = QGraphicsTextItem(f"HOMO (n={n_homo})")
        homo_text.setFont(text_font)
        homo_text.setDefaultTextColor(label_color)
        # Adjust text position to be above the line and left-aligned
        homo_text.setPos(level_line_x_start, y_homo - 22)
        self.scene.addItem(homo_text)

        # --- Draw LUMO Level ---
        y_lumo = energy_to_y(E_lumo_J)
        # Prevent drawing LUMO line directly on top of HOMO line if energies are identical
        if abs(y_lumo - y_homo) < 0.1:
            y_lumo -= 5 # Offset slightly if they overlap numerically after scaling

        self.scene.addLine(level_line_x_start, y_lumo, level_line_x_end, y_lumo, line_pen)
        lumo_text = QGraphicsTextItem(f"LUMO (n={n_lumo})")
        lumo_text.setFont(text_font)
        lumo_text.setDefaultTextColor(label_color)
        # Adjust text position to be above the line and left-aligned
        lumo_text.setPos(level_line_x_start, y_lumo - 22)
        self.scene.addItem(lumo_text)

        # --- Draw Transition Arrow ---
        # Only draw if levels are distinct and within drawable area
        if abs(y_homo - y_lumo) > 1:
            arrow_x = (level_line_x_start + level_line_x_end) / 2 # Center the arrow
            arrow_head_size = 8
            line_end_y = y_lumo + arrow_head_size if y_lumo < y_homo else y_lumo - arrow_head_size # Adjust based on direction

            # Main arrow line
            arrow_line = self.scene.addLine(arrow_x, y_homo, arrow_x, line_end_y, arrow_pen)

            # Arrowhead (pointing towards LUMO)
            angle = math.atan2(y_lumo - y_homo, 0) # Should be -pi/2 or pi/2
            p1 = QPointF(arrow_x, y_lumo)
            p2 = p1 - QPointF(math.sin(angle + math.pi / 3) * arrow_head_size,
                              math.cos(angle + math.pi / 3) * arrow_head_size)
            p3 = p1 - QPointF(math.sin(angle + math.pi - math.pi / 3) * arrow_head_size,
                              math.cos(angle + math.pi - math.pi / 3) * arrow_head_size)

            self.scene.addLine(QPointF(arrow_x, y_lumo), p2, arrow_pen)
            self.scene.addLine(QPointF(arrow_x, y_lumo), p3, arrow_pen)


            # Add Delta E label near arrow
            delta_e_text = QGraphicsTextItem(f"ΔE = {(E_lumo_J - E_homo_J) / E_CHARGE:.2f} eV")
            delta_e_text.setFont(text_font)
            delta_e_text.setDefaultTextColor(arrow_pen.color())
            # Position text to the right of the arrow, centered vertically
            text_rect = delta_e_text.boundingRect()
            delta_e_text.setPos(arrow_x + 10, (y_homo + y_lumo) / 2 - text_rect.height() / 2)
            self.scene.addItem(delta_e_text)

        # Set scene rect bounds based on drawing, adding padding
        self.scene.setSceneRect(-10, -10, scene_width + 20, scene_height + 20)


# --- Run the Application ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ConjugatedSystemSim()
    sys.exit(app.exec_())
