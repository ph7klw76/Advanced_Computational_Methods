
the program is built on PyQt5. To learn it click [here](https://www.pythonguis.com/pyqt5/)


![image](https://github.com/user-attachments/assets/39ffa1b0-2e2a-4ece-b97f-28d24dba837a)

```python
import sys
import math
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QSizePolicy)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF

# --- Constants ---
# Physical constants (mostly for reference, simulation uses scaled values)
FREQ_HZ = 2.45e9  # 2.45 GHz (Actual microwave frequency)
OMEGA_RAD_S = 2 * math.pi * FREQ_HZ # Actual angular frequency
DIPOLE_MOMENT_SI = 6.17e-30 # C*m (Actual dipole moment)

# Simulation parameters
NUM_MOLECULES = 50
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
MOLECULE_SIZE = 15  # Visual size of oxygen atom
BOND_LENGTH = MOLECULE_SIZE * 1.3
H_SIZE_FACTOR = 0.6 # Hydrogen atom size relative to oxygen

# --- Simulation Dynamics Parameters (Tunable) ---
SIM_UPDATE_MS = 20 # Update interval in milliseconds (50 FPS)
DT = SIM_UPDATE_MS / 1000.0 # Time step for simulation updates

# Scaled E-field parameters for visualization
SIM_OMEGA = 2 * math.pi * 1.0 # Visual oscillation frequency (e.g., 1 Hz)
SIM_E0_MAX = 50.0 # Maximum strength of the visual E-field (arbitrary units)

# Molecule behavior parameters (Tunable)
ALIGNMENT_STRENGTH = 50.0   # How strongly dipoles try to align with E-field
DAMPING_FACTOR = 0.90      # Reduces angular velocity (simulates friction)
BASE_JITTER_ANGLE = 0.1    # Base random rotation per step (radians)
BASE_JITTER_POS = 0.5      # Base random movement per step (pixels)
HEATING_FACTOR_INCREASE = 1.5 # How much jitter increases when microwave is on
TEMPERATURE_RAMP_RATE = 0.05 # How fast the "temperature factor" changes

# --- Water Molecule Class ---
class WaterMolecule:
    def __init__(self, x, y, angle_deg):
        self.pos = QPointF(x, y)
        self.angle = math.radians(angle_deg) # Angle of the dipole moment (O -> H midpoint) w.r.t positive x-axis
        self.angular_velocity = 0.0
        self.color_o = QColor(200, 0, 0) # Red for Oxygen
        self.color_h = QColor(200, 200, 200) # Light grey for Hydrogen
        self.dipole_strength = 1.0 # Scaled for simulation torque calculation

        # Calculate H positions relative to O based on angle
        # H-O-H angle is approx 104.5 degrees
        h_angle_offset = math.radians(104.5 / 2.0)
        self.h1_offset = QPointF(BOND_LENGTH * math.cos(h_angle_offset),
                                 BOND_LENGTH * math.sin(h_angle_offset))
        self.h2_offset = QPointF(BOND_LENGTH * math.cos(-h_angle_offset),
                                 BOND_LENGTH * math.sin(-h_angle_offset))

    def get_dipole_vector(self):
        # Dipole points from O towards midpoint between H's
        # For simplicity, aligned with the molecule's angle
        return QPointF(math.cos(self.angle), math.sin(self.angle))

    def update(self, dt, E_field_x, is_microwave_on, temperature_factor):
        # --- 1. Calculate Torque from E-field (Simplified Eq. 1) ---
        torque = 0.0
        if is_microwave_on and abs(E_field_x) > 1e-6:
            # Torque = mu x E = |mu| |E| sin(phi) * z_hat
            # phi is the angle between dipole and E-field (along x-axis)
            # Dipole vector = (cos(self.angle), sin(self.angle))
            # E vector = (E_field_x, 0)
            # Cross product magnitude in 2D: mu_x*E_y - mu_y*E_x
            dipole_vec = self.get_dipole_vector() * self.dipole_strength
            torque = dipole_vec.x() * 0 - dipole_vec.y() * E_field_x
            torque *= -ALIGNMENT_STRENGTH # Scale torque effect

        # --- 2. Apply Rotational Dynamics (Simplified Eq. 2) ---
        # Ignore inertia (I=1) for simplicity, apply torque directly to angular velocity
        # Add damping and stochastic torque (jitter)
        self.angular_velocity += torque * dt
        self.angular_velocity *= DAMPING_FACTOR # Rotational friction

        # Add thermal jitter (stochastic torque effect)
        random_torque = random.uniform(-1, 1) * BASE_JITTER_ANGLE * temperature_factor
        self.angular_velocity += random_torque * dt # Instantaneous random kick

        # Update angle
        self.angle += self.angular_velocity * dt
        self.angle = self.angle % (2 * math.pi) # Keep angle in [0, 2pi)

        # --- 3. Apply Translational Dynamics (Jitter) ---
        # Random walk representing thermal motion
        vx = random.uniform(-1, 1) * BASE_JITTER_POS * temperature_factor
        vy = random.uniform(-1, 1) * BASE_JITTER_POS * temperature_factor
        self.pos += QPointF(vx * dt * 50, vy * dt * 50) # Scale dt effect

        # --- 4. Boundary Conditions (Wrap around) ---
        if self.pos.x() < -MOLECULE_SIZE: self.pos.setX(WINDOW_WIDTH + MOLECULE_SIZE)
        if self.pos.x() > WINDOW_WIDTH + MOLECULE_SIZE: self.pos.setX(-MOLECULE_SIZE)
        if self.pos.y() < -MOLECULE_SIZE: self.pos.setY(WINDOW_HEIGHT + MOLECULE_SIZE)
        if self.pos.y() > WINDOW_HEIGHT + MOLECULE_SIZE: self.pos.setY(-MOLECULE_SIZE)


    def draw(self, painter):
        painter.save()
        painter.translate(self.pos)
        painter.rotate(math.degrees(self.angle)) # Rotate coordinate system

        # Draw Oxygen (at origin after translation)
        painter.setBrush(QBrush(self.color_o))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(0, 0), MOLECULE_SIZE, MOLECULE_SIZE)

        # Calculate Hydrogen positions in the rotated frame
        h1_pos = self.h1_offset
        h2_pos = self.h2_offset
        h_radius = MOLECULE_SIZE * H_SIZE_FACTOR

        # Draw Hydrogens
        painter.setBrush(QBrush(self.color_h))
        painter.drawEllipse(h1_pos, h_radius, h_radius)
        painter.drawEllipse(h2_pos, h_radius, h_radius)

        # Draw Bonds
        painter.setPen(QPen(Qt.black, 1))
        painter.drawLine(QPointF(0, 0), h1_pos)
        painter.drawLine(QPointF(0, 0), h2_pos)

        # Optional: Draw dipole moment vector (for debugging/visualization)
        # painter.setPen(QPen(Qt.blue, 2, Qt.DashLine))
        # painter.drawLine(QPointF(0,0), self.get_dipole_vector() * MOLECULE_SIZE * 1.5)

        painter.restore()

# --- Simulation Widget ---
class MicrowaveWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.molecules = []
        self.microwave_on = False
        self.simulation_time = 0.0
        self.current_E_field = 0.0
        self.temperature_factor = 1.0 # Represents thermal agitation level
        self.target_temperature_factor = 1.0

        self.init_molecules()
        self.init_timer()

    def init_molecules(self):
        self.molecules = []
        for _ in range(NUM_MOLECULES):
            x = random.uniform(MOLECULE_SIZE, WINDOW_WIDTH - MOLECULE_SIZE)
            y = random.uniform(MOLECULE_SIZE, WINDOW_HEIGHT - MOLECULE_SIZE)
            angle = random.uniform(0, 360)
            self.molecules.append(WaterMolecule(x, y, angle))

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(SIM_UPDATE_MS)

    def toggle_microwave(self, state):
        self.microwave_on = state
        if self.microwave_on:
            self.target_temperature_factor = HEATING_FACTOR_INCREASE
            print("Microwave ON")
        else:
            self.target_temperature_factor = 1.0 # Cool down to base level
            self.current_E_field = 0.0 # Ensure field is off immediately
            print("Microwave OFF")

    def update_simulation(self):
        self.simulation_time += DT

        # Update E-field based on microwave state
        if self.microwave_on:
            # E(t) = E0 * cos(omega*t) along x-axis
            self.current_E_field = SIM_E0_MAX * math.cos(SIM_OMEGA * self.simulation_time)
        else:
            self.current_E_field = 0.0

        # Smoothly adjust temperature factor towards target
        if abs(self.temperature_factor - self.target_temperature_factor) > 0.01:
             diff = self.target_temperature_factor - self.temperature_factor
             self.temperature_factor += diff * TEMPERATURE_RAMP_RATE
        else:
            self.temperature_factor = self.target_temperature_factor


        # Update each molecule
        for molecule in self.molecules:
            molecule.update(DT, self.current_E_field, self.microwave_on, self.temperature_factor)

        # Trigger repaint
        self.update()

    # --- CORRECTED paintEvent Method ---
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(240, 240, 255)) # Light blue background

        # Draw Molecules
        for molecule in self.molecules:
            molecule.draw(painter)

        # Draw E-field indicator (optional)
        if self.microwave_on:
            arrow_length = self.current_E_field * 2 # Scale for visibility
            arrow_y = 30 # Keep Y as integer is fine
            painter.setPen(QPen(QColor(0, 150, 0), 2))
            painter.setBrush(QBrush(QColor(0, 150, 0)))

            # Base line - Use integer casting just in case (though // should make it int)
            base_start_x = WINDOW_WIDTH // 2 - 100
            base_end_x = WINDOW_WIDTH // 2 + 100
            painter.drawLine(int(base_start_x), int(arrow_y), int(base_end_x), int(arrow_y))

            # Arrow head indicating direction and strength
            start_x = WINDOW_WIDTH // 2 # This is likely an int or float needing casting
            end_x = start_x + arrow_length # end_x is definitely float

            # *** FIX: Cast coordinates to integers ***
            painter.drawLine(int(start_x), int(arrow_y), int(end_x), int(arrow_y))

            # Arrow tips
            if abs(arrow_length) > 1:
                direction = 1 if arrow_length > 0 else -1
                tip_len = 8
                tip_angle = math.radians(20)

                # *** FIX: Cast coordinates to integers ***
                tip1_x = int(end_x - direction * tip_len * math.cos(tip_angle))
                tip1_y = int(arrow_y + tip_len * math.sin(tip_angle))
                tip2_x = int(end_x - direction * tip_len * math.cos(tip_angle)) # x is same for both tips
                tip2_y = int(arrow_y - tip_len * math.sin(tip_angle)) # y-coordinate for the other tip line

                # Draw the two lines forming the arrowhead
                painter.drawLine(int(end_x), int(arrow_y), tip1_x, tip1_y)
                painter.drawLine(int(end_x), int(arrow_y), tip2_x, tip2_y) # Use tip2_x, tip2_y

            painter.setFont(QFont("Arial", 10))
            # Position text relative to the base line start
            painter.drawText(base_start_x, arrow_y - 10, "E-Field (x-axis)")


        # Draw Status Text
        painter.setPen(Qt.black)
        painter.setFont(QFont("Arial", 12))
        status = "Microwave: ON" if self.microwave_on else "Microwave: OFF"
        painter.drawText(10, 20, status)
        painter.drawText(10, 40, f"Agitation Factor: {self.temperature_factor:.2f}")
    # --- End of corrected paintEvent ---


# --- Main Application Window ---
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microwave Water Heating Simulation")
        # Make window slightly larger vertically to accommodate button without overlap
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT + 60) # Extra height for button

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Simulation Area
        self.simulation_widget = MicrowaveWidget()
        # Make sure the widget takes up available space
        self.simulation_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.simulation_widget)

        # Control Button
        self.toggle_button = QPushButton("Turn Microwave ON")
        self.toggle_button.setCheckable(True) # Make it a toggle button
        self.toggle_button.clicked.connect(self.on_toggle_button)
        self.layout.addWidget(self.toggle_button) # Add button below simulation

    def on_toggle_button(self, checked):
        self.simulation_widget.toggle_microwave(checked)
        if checked:
            self.toggle_button.setText("Turn Microwave OFF")
        else:
            self.toggle_button.setText("Turn Microwave ON")


# --- Run the Application ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
```
