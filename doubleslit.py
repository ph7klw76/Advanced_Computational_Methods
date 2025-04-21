import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QLabel, QSlider, QSizePolicy, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtGui import QPainter, QColor, QPen, QFont

# Matplotlib imports
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Constants and Simulation Parameters ---
PLANCK_CONST_REL = 1.0
ELECTRON_MASS_REL = 1.0
SLIT_SCREEN_DISTANCE = 100.0 # Arbitrary distance L (affects interference calculation)

SCREEN_WIDTH_UNITS = 100.0 # Width of the detection screen in spatial units
SCREEN_BINS = 200

# --- Helper Functions (same as before) ---
def calculate_wavelength(energy):
    if energy <= 0: return float('inf')
    momentum_sq = 2 * ELECTRON_MASS_REL * energy
    return PLANCK_CONST_REL / np.sqrt(momentum_sq)

def interference_probability(y_positions, wavelength, slit_separation, slit_screen_dist):
    if wavelength == float('inf') or wavelength == 0:
        return np.ones_like(y_positions) / len(y_positions)
    k = 2 * np.pi / wavelength
    path_difference = slit_separation * y_positions / slit_screen_dist
    phase_difference = k * path_difference
    probability = np.cos(phase_difference / 2)**2 + 0.01
    total_prob = np.sum(probability)
    return probability / total_prob if total_prob > 0 else np.ones_like(y_positions) / len(y_positions)

def single_slit_probability(y_positions, center_y, slit_width_factor=8.0): # Increased factor for narrower peaks
    variance = (SCREEN_WIDTH_UNITS / slit_width_factor)**2
    exponent = - (y_positions - center_y)**2 / (2 * variance)
    probability = np.exp(exponent) + 0.001 # Small baseline
    total_prob = np.sum(probability)
    return probability / total_prob if total_prob > 0 else np.ones_like(y_positions) / len(y_positions)


# --- Simulation Thread (same as before) ---
class SimulationThread(QThread):
    electron_detected = pyqtSignal(int)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent_widget = parent
        self.running = False
        self.detector_active = False
        self.energy = 1.0
        self.slit_separation = 5.0
        self.y_positions = np.linspace(-SCREEN_WIDTH_UNITS / 2, SCREEN_WIDTH_UNITS / 2, SCREEN_BINS)
        self._stop_requested = False

    def run(self):
        self.running = True
        self._stop_requested = False
        while not self._stop_requested:
            if not self.parent_widget.continuous_mode and self._stop_requested: break # Single shot check

            bin_index = self.simulate_one_electron()
            if bin_index is not None:
                self.electron_detected.emit(bin_index)

            if not self.parent_widget.continuous_mode: break
            else: self.msleep(5) # Small delay for continuous mode

        self.running = False
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def simulate_one_electron(self):
        slit_pos_1 = -self.slit_separation / 2
        slit_pos_2 = +self.slit_separation / 2

        if self.detector_active:
            # Particle Behavior
            prob_dist_1 = single_slit_probability(self.y_positions, slit_pos_1)
            prob_dist_2 = single_slit_probability(self.y_positions, slit_pos_2)
            # Combine probabilities (representing two separate possibilities)
            # We choose a slit first, then sample from its distribution
            if np.random.rand() < 0.5: # Choose slit 1
                 chosen_bin = np.random.choice(SCREEN_BINS, p=prob_dist_1)
            else: # Choose slit 2
                 chosen_bin = np.random.choice(SCREEN_BINS, p=prob_dist_2)
            return chosen_bin
        else:
            # Wave Behavior
            wavelength = calculate_wavelength(self.energy)
            prob_dist = interference_probability(
                self.y_positions, wavelength, self.slit_separation, SLIT_SCREEN_DISTANCE
            )
            try:
                prob_dist /= np.sum(prob_dist) # Ensure normalization
                chosen_bin = np.random.choice(SCREEN_BINS, p=prob_dist)
                return chosen_bin
            except ValueError as e:
                 print(f"Error choosing bin (wave): {e}, sum={np.sum(prob_dist)}")
                 return None # Skip if probability is invalid


# --- Matplotlib Canvas Widget (same as before) ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#E0E0E0') # Light gray background
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#F0F0F0') # Slightly lighter plot area
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.fig.tight_layout()


# --- Experiment Diagram Widget ---
class ExperimentDiagram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120) # Ensure enough space for drawing
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.detector_active = False
        self.slit_separation_rel = 0.2 # Relative separation (0 to 1)
        self.slit_width_rel = 0.05    # Relative width

    def set_detector_active(self, active):
        if self.detector_active != active:
            self.detector_active = active
            self.update() # Trigger repaint

    def set_slit_params(self, separation_value, separation_max, width_value=1):
        # separation_value is raw slider value, max is slider max
        # We map this to a relative visual separation for drawing
        self.slit_separation_rel = separation_value / separation_max * 0.4 + 0.1 # Map to range [0.1, 0.5] approx
        # Slit width could also be adjustable, but fixed for now
        self.slit_width_rel = max(0.02, 0.1 - separation_value / separation_max * 0.08) # Narrower slits as separation increases
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        mid_y = height / 2

        # Colors
        bg_color = QColor("#D0D0D0") # Background for the diagram area
        barrier_color = QColor("#505050")
        screen_color = QColor("#707070")
        source_color = QColor("#FF6347") # Tomato red
        detector_color = QColor("#4682B4") # Steel blue
        electron_wave_color = QColor(source_color)
        electron_wave_color.setAlpha(80) # Transparent wave

        painter.fillRect(self.rect(), bg_color)

        # --- Draw Components ---
        barrier_x = width * 0.4
        barrier_width = 10
        screen_x = width * 0.85
        screen_width = 10
        source_x = width * 0.1
        source_radius = 5

        # Source
        painter.setBrush(source_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(source_x - source_radius), int(mid_y - source_radius), int(2 * source_radius), int(2 * source_radius))

        # Barrier
        painter.setBrush(barrier_color)
        barrier_rect = QRectF(barrier_x, 0, barrier_width, height)
        painter.drawRect(barrier_rect)

        # Slits (Calculate positions based on relative separation and width)
        slit_half_sep_pixels = self.slit_separation_rel * height / 2
        slit_half_width_pixels = self.slit_width_rel * height / 2

        slit1_top = mid_y - slit_half_sep_pixels - slit_half_width_pixels
        slit1_bottom = mid_y - slit_half_sep_pixels + slit_half_width_pixels
        slit2_top = mid_y + slit_half_sep_pixels - slit_half_width_pixels
        slit2_bottom = mid_y + slit_half_sep_pixels + slit_half_width_pixels

        painter.setBrush(bg_color) # "Cut out" the slits
        painter.drawRect(QRectF(barrier_x, slit1_top, barrier_width, slit1_bottom - slit1_top))
        painter.drawRect(QRectF(barrier_x, slit2_top, barrier_width, slit2_bottom - slit2_top))

        # Screen
        painter.setBrush(screen_color)
        painter.drawRect(int(screen_x), 0, screen_width, height)

        # Electron Path Representation
        painter.setPen(QPen(electron_wave_color, 2, Qt.DashLine))
        painter.drawLine(int(source_x + source_radius), int(mid_y), int(barrier_x), int(slit1_top + slit_half_width_pixels))
        painter.drawLine(int(source_x + source_radius), int(mid_y), int(barrier_x), int(slit2_top + slit_half_width_pixels))

        # Detector (if active)
        if self.detector_active:
            detector_radius = 8
            painter.setBrush(detector_color)
            painter.setPen(QPen(Qt.black, 1))
            # Draw detector near slit 1
            painter.drawEllipse(int(barrier_x + barrier_width + 5), int(slit1_top + slit_half_width_pixels - detector_radius), int(2*detector_radius), int(2*detector_radius))
             # Draw detector near slit 2
            painter.drawEllipse(int(barrier_x + barrier_width + 5), int(slit2_top + slit_half_width_pixels - detector_radius), int(2*detector_radius), int(2*detector_radius))
            # Label
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 8))
            painter.drawText(int(barrier_x + barrier_width + 5 + 2 * detector_radius + 2), int(mid_y + 5), "Detectors ON")


        # Labels
        painter.setPen(Qt.black)
        painter.setFont(QFont("Arial", 9))
        painter.drawText(int(source_x-15), int(mid_y - 15), "Source")
        painter.drawText(int(barrier_x - 15), int(height - 10), "Barrier")
        painter.drawText(int(screen_x - 15), int(height - 10), "Screen")

        painter.end()


# --- Main Application Window ---
class DoubleSlitSim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double-Slit Experiment Simulator")
        self.setGeometry(100, 100, 900, 750) # Increased size slightly

        # --- Central Widget and Layout ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(10) # Add spacing between elements

        # --- Simulation State ---
        self.screen_hits = np.zeros(SCREEN_BINS, dtype=int)
        self.electron_count = 0
        self.continuous_mode = False

        # --- Setup UI Elements ---
        self._setup_diagram() # NEW
        self._setup_plot()
        self._setup_controls() # Modified layout
        self._connect_signals()

        # --- Simulation Thread ---
        self.sim_thread = SimulationThread(self)
        self.sim_thread.electron_detected.connect(self.record_detection, Qt.QueuedConnection)

        # Initial parameter update and drawing
        self._update_parameters()
        self._update_plot()

    def _setup_diagram(self):
        """Creates the experiment diagram widget."""
        self.diagram_widget = ExperimentDiagram(self)
        self.layout.addWidget(self.diagram_widget)

    def _setup_plot(self):
        """Creates the Matplotlib canvas."""
        plot_group = QGroupBox("Detection Screen Pattern")
        plot_layout = QVBoxLayout()
        self.plot_canvas = MplCanvas(self, width=7, height=4, dpi=100)
        plot_layout.addWidget(self.plot_canvas)
        plot_group.setLayout(plot_layout)
        self.layout.addWidget(plot_group)

        self.bin_centers = np.linspace(-SCREEN_WIDTH_UNITS / 2, SCREEN_WIDTH_UNITS / 2, SCREEN_BINS)

    def _setup_controls(self):
        """Creates the control widgets using GroupBoxes."""
        controls_layout = QHBoxLayout() # Main layout for control sections

        # --- Simulation Control Group ---
        sim_control_group = QGroupBox("Simulation Control")
        sim_control_vbox = QVBoxLayout()
        self.fire_button = QPushButton("Fire Single Electron")
        self.run_button = QPushButton("Run Continuously")
        self.stop_button = QPushButton("Stop Continuous")
        self.reset_button = QPushButton("Reset Simulation")
        self.stop_button.setEnabled(False)
        sim_control_vbox.addWidget(self.fire_button)
        sim_control_vbox.addWidget(self.run_button)
        sim_control_vbox.addWidget(self.stop_button)
        sim_control_vbox.addWidget(self.reset_button)
        sim_control_vbox.addStretch()
        sim_control_group.setLayout(sim_control_vbox)
        controls_layout.addWidget(sim_control_group)

        # --- Parameters Group ---
        params_group = QGroupBox("Experiment Parameters")
        params_vbox = QVBoxLayout()

        self.detector_checkbox = QCheckBox("Activate Slit Detector")
        params_vbox.addWidget(self.detector_checkbox)
        params_vbox.addSpacing(10)

        # Energy Slider
        energy_label = QLabel("Electron Energy (Higher -> Shorter Wavelength)")
        self.energy_slider = QSlider(Qt.Horizontal)
        self.energy_slider.setMinimum(1)
        self.energy_slider.setMaximum(100)
        self.energy_slider.setValue(20)
        self.energy_value_label = QLabel(f"Value: {self.energy_slider.value() / 10.0:.1f}")
        params_vbox.addWidget(energy_label)
        energy_hbox = QHBoxLayout()
        energy_hbox.addWidget(self.energy_slider)
        energy_hbox.addWidget(self.energy_value_label)
        params_vbox.addLayout(energy_hbox)
        params_vbox.addSpacing(10)

        # Separation Slider
        separation_label = QLabel("Slit Separation (Wider -> Finer Interference)")
        self.separation_slider = QSlider(Qt.Horizontal)
        self.separation_slider.setMinimum(1)
        self.separation_slider.setMaximum(50)
        self.separation_slider.setValue(10)
        self.separation_value_label = QLabel(f"Value: {self.separation_slider.value() / 2.0:.1f}")
        params_vbox.addWidget(separation_label)
        separation_hbox = QHBoxLayout()
        separation_hbox.addWidget(self.separation_slider)
        separation_hbox.addWidget(self.separation_value_label)
        params_vbox.addLayout(separation_hbox)

        params_vbox.addStretch()
        params_group.setLayout(params_vbox)
        controls_layout.addWidget(params_group)

        # --- Status Group ---
        status_group = QGroupBox("Status")
        status_vbox = QVBoxLayout()
        self.status_label = QLabel("Status: Idle")
        self.count_label = QLabel("Electrons Detected: 0")
        self.mode_label = QLabel("Mode: Wave (Detector OFF)")
        status_vbox.addWidget(self.status_label)
        status_vbox.addWidget(self.count_label)
        status_vbox.addWidget(self.mode_label)
        status_vbox.addStretch()
        status_group.setLayout(status_vbox)
        controls_layout.addWidget(status_group)

        self.layout.addLayout(controls_layout) # Add the controls HBox to the main VBox


    def _connect_signals(self):
        """Connects GUI signals to slots."""
        self.fire_button.clicked.connect(self.fire_single_electron)
        self.run_button.clicked.connect(self.run_continuously)
        self.stop_button.clicked.connect(self.stop_continuous)
        self.reset_button.clicked.connect(self.reset_simulation)

        # Update parameters AND the diagram
        self.detector_checkbox.stateChanged.connect(self._update_parameters)
        self.energy_slider.valueChanged.connect(self._update_parameters)
        self.separation_slider.valueChanged.connect(self._update_parameters)


    def _update_parameters(self):
        """Reads parameters, updates thread, labels, AND diagram."""
        detector_on = self.detector_checkbox.isChecked()
        energy = self.energy_slider.value() / 10.0
        separation = self.separation_slider.value() / 2.0
        separation_raw = self.separation_slider.value() # For diagram scaling
        separation_max = self.separation_slider.maximum() # For diagram scaling


        self.energy_value_label.setText(f"Value: {energy:.1f}")
        self.separation_value_label.setText(f"Value: {separation:.1f}")

        mode_text = "Mode: Particle (Detector ON)" if detector_on else "Mode: Wave (Detector OFF)"
        self.mode_label.setText(mode_text)

        # Update simulation thread
        self.sim_thread.detector_active = detector_on
        self.sim_thread.energy = energy
        self.sim_thread.slit_separation = separation

        # Update diagram widget
        self.diagram_widget.set_detector_active(detector_on)
        self.diagram_widget.set_slit_params(separation_raw, separation_max)

        # Optional: Reset if parameters change? Decide based on desired behavior.
        # self.reset_simulation()


    def fire_single_electron(self):
        """Handles 'Fire Single Electron'."""
        if not self.sim_thread.isRunning():
            self.status_label.setText("Status: Firing...")
            self.continuous_mode = False
            # Disable buttons during single fire
            self.fire_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.sim_thread.start()
        else: print("Simulation already running.")

    def run_continuously(self):
        """Handles 'Run Continuously'."""
        if not self.sim_thread.isRunning():
            self.continuous_mode = True
            self.run_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.fire_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.status_label.setText("Status: Running...")
            self.sim_thread.start()
        else: print("Simulation already running.")

    def stop_continuous(self):
        """Handles 'Stop Continuous'."""
        if self.sim_thread.isRunning() and self.continuous_mode:
            self.status_label.setText("Status: Stopping...")
            self.sim_thread.stop()
            # Buttons will be re-enabled when the thread finishes cleanly via record_detection or timeout
        else: # Handle stopping after single fire if needed (though usually stops itself)
             if self.sim_thread.isRunning():
                 self.sim_thread.stop()


    def record_detection(self, bin_index):
        """Records hit, updates plot, and handles state after run."""
        if 0 <= bin_index < SCREEN_BINS:
            self.screen_hits[bin_index] += 1
            self.electron_count += 1
            self.count_label.setText(f"Electrons Detected: {self.electron_count}")

            # Update plot (less frequent in continuous mode for performance)
            update_freq = 1 if not self.continuous_mode else 20 # Update every hit in single, every 20 in continuous
            if self.electron_count % update_freq == 0:
                self._update_plot()

        # Check if the thread should stop / has stopped
        if not self.sim_thread.isRunning():
             # This means the thread finished its loop (either one shot or was stopped)
             QTimer.singleShot(10, self._finalize_stop) # Use QTimer to ensure this runs after signal processing

    def _finalize_stop(self):
        """Update UI elements after thread has confirmed stopped."""
        # Ensure buttons are correctly enabled/disabled only when thread is truly stopped
        if not self.sim_thread.isRunning():
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.fire_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            if not self.continuous_mode: # Keep "Stopping..." if user pressed stop
                 self.status_label.setText("Status: Idle")
            self.continuous_mode = False # Ensure continuous mode is off

    def _update_plot(self):
        """Updates the Matplotlib histogram display with better visuals."""
        ax = self.plot_canvas.axes
        ax.clear()

        bar_width = self.bin_centers[1] - self.bin_centers[0]
        ax.bar(self.bin_centers, self.screen_hits, width=bar_width, align='center',
               color='cornflowerblue', edgecolor='grey', alpha=0.8) # Nicer color, slight transparency

        ax.set_xlabel("Position on Screen (y)")
        ax.set_ylabel("Number of Electrons Detected")
        # Title moved to GroupBox
        ax.set_xlim(self.bin_centers[0] - bar_width/2, self.bin_centers[-1] + bar_width/2)
        if np.max(self.screen_hits) > 0:
             ax.set_ylim(bottom=0, top=np.max(self.screen_hits) * 1.15) # Slightly more headroom
        else:
             ax.set_ylim(bottom=0, top=10)

        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_facecolor('#F0F0F0') # Match canvas facecolor

        # Optional: Slit position lines (can be less prominent now diagram exists)
        # slit_sep = self.separation_slider.value() / 2.0
        # ax.axvline(-slit_sep / 2, color='r', linestyle=':', linewidth=0.8, alpha=0.7)
        # ax.axvline( slit_sep / 2, color='g', linestyle=':', linewidth=0.8, alpha=0.7)

        self.plot_canvas.fig.tight_layout() # Re-apply tight layout
        self.plot_canvas.draw()


    def reset_simulation(self):
        """Resets the simulation state."""
        if self.sim_thread.isRunning():
            print("Cannot reset while running. Stop first.")
            return
        self.screen_hits.fill(0)
        self.electron_count = 0
        self.count_label.setText("Electrons Detected: 0")
        self.status_label.setText("Status: Idle")
        self._update_plot()
        # Ensure buttons are in correct state after reset
        self._finalize_stop()
        print("Simulation Reset.")

    def closeEvent(self, event):
        """Ensure thread stopped on close."""
        if self.sim_thread.isRunning():
            self.sim_thread.stop()
            self.sim_thread.wait(500) # Wait max 500ms
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        import qdarkstyle
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    except ImportError:
        app.setStyle("Fusion") # Use Fusion style if dark style unavailable

    main_window = DoubleSlitSim()
    main_window.show()
    sys.exit(app.exec_())
