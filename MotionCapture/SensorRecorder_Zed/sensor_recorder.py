import math
import numpy as np
import time
import datetime
import pickle

#osc
from pythonosc import dispatcher
from pythonosc import osc_server

#gui
from PyQt5 import QtWidgets, QtCore
from vispy import scene
from vispy.app import use_app, Timer
from vispy.scene import SceneCanvas, visuals


"""
OSC Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

"""
OSC Receiver
"""

class OscReceiver(QtCore.QObject):
    
    new_data = QtCore.pyqtSignal(dict)
    
    def __init__(self, ip, port, parent=None):
        super().__init__(parent=parent)
        
        self.ip = ip
        self.port = port
        
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/*", self.receive)
        self.server = osc_server.BlockingOSCUDPServer((self.ip, self.port), self.dispatcher)
        
    def start(self):
        print("OscReceiver start")
        self.server.serve_forever()
    
    def stop(self):
        print("OscReceiver stop")
        self.server.shutdown()
    
    def receive(self, addr, *args):
        
        osc_address = addr
        osc_values = args
        
        #print("osc_address ", osc_address, " osc_values ", osc_values)
        
        values_dict = {
            osc_address: osc_values
            }
        
        self.new_data.emit(values_dict)

"""
Motion Recorder
"""

class MotionRecorder(QtCore.QObject):
    
    time_data = QtCore.pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        
        self.recording = None
        self.recording_active = False
        self.recording_start_time = None
        
        self._init_recording(0)
        
    def _init_recording(self, class_id):
        
        self.recording = {"class_id": class_id,
                          "time_stamps": [],
                          "sensor_ids": [],
                          "sensor_values": []
                          }
        
    def _save_recording(self):

        with open("recordings/Mocap_class_{}_time_{}.pkl".format(self.recording["class_id"], self.recording_start_time), 'wb') as f:
            pickle.dump(self.recording, f)
            
    def set_class_id(self, class_id):
        
        if self.recording_active == True:
            return
        
        print("set_class_id ", class_id)
        
        self._init_recording(class_id)
        
    def toggle_record(self):
        
        if self.recording_active == True:
            self.stop_record()
        else:
            self.start_record()
    
    def start_record(self):
        
        if self.recording_active == True:
            return
        
        print("start_record ")
        
        self.recording_active = True
        self.recording_start_time = time.time()
        
    def stop_record(self):
        
        if self.recording_active == False:
            return
        
        print("stop_record ")
        
        self.recording_active = False
        self._save_recording()
        
    def record(self, sensor_data):
        
        if self.recording_active == False:
            return
        
        time_stamp = time.time() - self.recording_start_time
        sensor_id = list(sensor_data.keys())[0]
        sensor_value = list(sensor_data.values())[0]
        
        self.recording["time_stamps"].append(time_stamp)  
        self.recording["sensor_ids"].append(sensor_id)  
        self.recording["sensor_values"].append(sensor_value)
        
        self.time_data.emit(time_stamp)
    
"""
GUI
"""

class BarView:
    def __init__(self, max_value_count, colors, parent_view=None):
        self.max_value_count = max_value_count
        self.value_count = max_value_count
        self.colors = colors
        self.parent_view = parent_view

        self.bar_width = 1.0 / self.value_count
        self.bar_centers_x = np.linspace(self.bar_width / 2, 1.0 - self.bar_width / 2, self.value_count)

        # 4 vertices per rectangle (bar), creating proper faces
        self.vertices = np.zeros((self.value_count * 4, 3), dtype=np.float32)
        self.colors_arr = np.zeros((self.value_count * 4, 4), dtype=np.float32)

        # Faces: Two triangles per rectangle (each face references 3 vertices)
        self.faces = np.zeros((self.value_count * 2, 3), dtype=np.uint32)
        
        # Generate faces for rectangles
        for i in range(self.value_count):
            base_face = i * 2
            base_vert = i * 4
            # Triangle 1: bottom-left, bottom-right, top-right
            self.faces[base_face] = [base_vert, base_vert + 1, base_vert + 2]
            # Triangle 2: bottom-left, top-right, top-left
            self.faces[base_face + 1] = [base_vert, base_vert + 2, base_vert + 3]

        self._initialize_geometry()

        # Create mesh with explicit faces
        self.mesh = visuals.Mesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_colors=self.colors_arr,
            mode='triangles',
            parent=self.parent_view
        )

    def _initialize_geometry(self):
        """Initialize vertex positions and colors for all bars"""
        half_width = self.bar_width / 2

        for i, center_x in enumerate(self.bar_centers_x):
            base_index = i * 4
            bottom_y = 0.0
            top_y = 0.01

            # Rectangle corners (4 vertices per bar)
            bl = (center_x - half_width, bottom_y, 0)  # bottom-left
            br = (center_x + half_width, bottom_y, 0)  # bottom-right
            tr = (center_x + half_width, top_y, 0)     # top-right
            tl = (center_x - half_width, top_y, 0)     # top-left

            self.vertices[base_index] = bl
            self.vertices[base_index + 1] = br
            self.vertices[base_index + 2] = tr
            self.vertices[base_index + 3] = tl

            # Set colors for all 4 vertices of this bar
            color = self.colors[i] if i < len(self.colors) else (1, 1, 1, 0.8)
            self.colors_arr[base_index:base_index+4] = color

    def resetValueCount(self, value_count):
        """Optimize: only recalculate if count actually changed"""
        if value_count == self.value_count:
            return

        self.value_count = value_count
        self.bar_width = 0.9 / self.value_count
        self.bar_centers_x = np.linspace(self.bar_width / 2, 1.0 - self.bar_width / 2, self.value_count)
        
        # Recreate faces for new bar count
        self.faces = np.zeros((self.value_count * 2, 3), dtype=np.uint32)
        for i in range(self.value_count):
            base_face = i * 2
            base_vert = i * 4
            self.faces[base_face] = [base_vert, base_vert + 1, base_vert + 2]
            self.faces[base_face + 1] = [base_vert, base_vert + 2, base_vert + 3]
        
        self._initialize_geometry()

    def update(self, values):

        """Batch update all bar heights"""
        value_count = min(len(values), self.max_value_count)

        if value_count != self.value_count:
            self.resetValueCount(value_count)

        half_width = self.bar_width / 2

        # Update vertex positions for all bars
        for i in range(self.value_count):
            base_index = i * 4
            center_x = self.bar_centers_x[i]
            value = values[i] if i < len(values) else 0

            #bottom_y = 0
            bottom_y = min(value, -0.0001)   # Ensure minimum height
            #top_y = max(abs(value), 0.0001)  # Ensure minimum height
            top_y = max(value, 0.0001)  # Ensure minimum height

            # Rectangle corners
            bl = (center_x - half_width, bottom_y, 0)
            br = (center_x + half_width, bottom_y, 0)
            tr = (center_x + half_width, top_y, 0)
            tl = (center_x - half_width, top_y, 0)

            self.vertices[base_index] = bl
            self.vertices[base_index + 1] = br
            self.vertices[base_index + 2] = tr
            self.vertices[base_index + 3] = tl

        # Single GPU update for all bars
        self.mesh.set_data(vertices=self.vertices, faces=self.faces, vertex_colors=self.colors_arr)

class TimePlotView:
    def __init__(self, value_count, time_count, colors, parent_view=None):
        self.value_count = value_count
        self.time_count = time_count
        self.colors = colors
        self.parent_view = parent_view

        # Store values
        self.values = np.zeros((self.time_count, self.value_count))

        # Time axis for all points
        self.time_line = np.linspace(0.0, 1.0, self.time_count)

        # Preallocate all line visuals (one per trace, or batch all into one if desired)
        self.lines = []
        for i in range(self.value_count):
            color = colors[i]
            line = visuals.Line(pos=np.zeros((self.time_count, 2)),
                                color=color,
                                width=2,
                                method='gl',
                                parent=self.parent_view)
            self.lines.append(line)

    def update(self, values):
        self.values = np.roll(self.values, -1, axis=0)
        self.values[-1] = values

        for i in range(self.value_count):
            # Update all points for the i-th trace
            pos = np.column_stack((self.time_line, self.values[:, i]))
            self.lines[i].set_data(pos=pos)


class SensorView:
    def __init__(self, title, value_dim, value_range, time_steps, colors):
        self.canvas = SceneCanvas()
        self.grid = self.canvas.central_widget.add_grid()
        
        
        title = scene.Label(title, color='white')
        title.height_max = 40
        self.grid.add_widget(title, row=0, col=1, col_span=2)

        yaxis = scene.AxisWidget(orientation='left',
                                 axis_font_size=12,
                                 axis_label_margin=50,
                                 tick_label_margin=5)
        yaxis.width_max = 80
        self.grid.add_widget(yaxis, row=1, col=0)
        

        self.bar_view = self.grid.add_view(1, 1, bgcolor="white")
        self.plot_right = self.grid.add_view(1, 2, bgcolor="white")
        
        self.bar_view.width_max = 50
        
        self.bars = BarView(value_dim, colors, self.bar_view.scene)
        self.plots = TimePlotView(value_dim, time_steps, colors, self.plot_right.scene)

        self.bar_view.camera = "panzoom"
        self.bar_view.camera.set_range(x=(0.0, 1.0), y=(value_range[0], value_range[1]))
        
        self.plot_right.camera = "panzoom"
        self.plot_right.camera.set_range(x=(0.0, 1.0), y=(value_range[0], value_range[1]))
        
        yaxis.link_view(self.plot_right)
        
    def update_data(self, data):
        self.bars.update(data)
        self.plots.update(data)

class Canvas:
    def __init__(self, size):
        self.size = size
        self.canvas = SceneCanvas(size = size, keys="interactive")
        self.grid = self.canvas.central_widget.add_grid()
        self.views = {}
        
        self.canvas.measure_fps(window=1, callback='%1.1f FPS')
        
    def add_sensor_view(self, name, value_dim, value_range, time_steps, colors):
        sensor_view = SensorView(name, value_dim, value_range, time_steps, colors)
        self.grid.add_widget(sensor_view.canvas.central_widget, len(self.views), 0)
        self.views[name] = sensor_view

    def update_data(self, new_data):
        
        key = list(new_data.keys())[0]
        value = list(new_data.values())[0]
        
        #print("key ", key, " value ", value)
        
        if key in self.views:
            self.views[key].update_data(value)
            
class MainWindow(QtWidgets.QMainWindow):
    
    closing = QtCore.pyqtSignal()
    
    def __init__(self, canvas, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setWindowTitle("Mocap Recorder")
        
        # main layout
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        
        #self._controls = Controls()
        #main_layout.addWidget(self._controls)
        self.canvas = canvas
        main_layout.addWidget(self.canvas.canvas.native)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # controls layout
        controls_layout = QtWidgets.QHBoxLayout()
        
        self.class_label = QtWidgets.QLabel("class:", self)
        self.class_input = QtWidgets.QSpinBox(self)
        self.start_buttom = QtWidgets.QPushButton("start", self)
        self.stop_buttom = QtWidgets.QPushButton("stop", self)
        self.time_label = QtWidgets.QLabel("", self)
        
        self.class_label.setFixedWidth(30)
        self.class_input.setFixedWidth(80)
        self.start_buttom.setFixedWidth(80)
        self.stop_buttom.setFixedWidth(80)
        #self.time_label.setFixedWidth(200)
        
        controls_layout.addWidget(self.class_label)
        controls_layout.addWidget(self.class_input)
        controls_layout.addWidget(self.start_buttom)
        controls_layout.addWidget(self.stop_buttom)
        controls_layout.addWidget(self.time_label)
        
        main_layout.addLayout(controls_layout)
        
    def showTime(self, time):

        time_text = str(datetime.timedelta(milliseconds=int(time * 1000.0)))
        self.time_label.setText(f"Recording Time: {time_text}")

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)

if __name__ == "__main__":
    
    # osc
    osc_record_active = False
    
    osc_receiver = OscReceiver(osc_receive_ip, osc_receive_port)
    motion_recorder = MotionRecorder()
    
    #gui
    app = use_app("pyqt5")
    app.create()
    
    canvas = Canvas((800, 600))
    
    """
    #Example 1: visualise joint positions from a mocap recording with 29 joints, and 3 dimensions for each position, timeseries contain 100 values and are all coloured red
    canvas.add_sensor_view("/mocap/0/joint/pos_world", 87, (-2000.0, 2000.0), 100, [(1.0, 0.0, 0.0, 1.0)] * 87)
    """

    """
    #Example 2: visualise joint positions from a pose estimation recording with 17 joints, and 2 dimensions for each position, timeseries contain 100 values and are all coloured red
    canvas.add_sensor_view("/mocap/0/joint/pos_world", 87, (-2000.0, 2000.0), 100, [(1.0, 0.0, 0.0, 1.0)] * 87)
    """

    """
    #Example 3: visualise 3 gyroscope values from a single IMU sensor, timeseries contain 100 values and are coloured red, green, and blue
    canvas.add_sensor_view("/gyroscope", 3, (-50.0, 50.0), 100, ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)))
    """

    """
    #Example 3: visualise 3 gyroscope values from a single IMU sensor, timeseries contain 100 values and are coloured red, green, and blue
    canvas.add_sensor_view("/gyroscope", 3, (-50.0, 50.0), 100, ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)))
    """


    #Example 4: visualise 3 accelerometer values for a single IMU sensor, timeseries contain 100 valuees and are coloured red, green, and blue
    canvas.add_sensor_view("/accelerometer", 3, (-50.0, 50.0), 100, ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)))

    win = MainWindow(canvas)

    win.class_input.valueChanged.connect(motion_recorder.set_class_id)
    win.start_buttom.clicked.connect(motion_recorder.start_record)
    win.stop_buttom.clicked.connect(motion_recorder.stop_record)
    motion_recorder.time_data.connect(win.showTime)

    osc_thread = QtCore.QThread(parent=win)
    osc_receiver.moveToThread(osc_thread)
    
    osc_receiver.new_data.connect(motion_recorder.record)
    osc_receiver.new_data.connect(canvas.update_data)
    osc_thread.started.connect(osc_receiver.start)
    
    win.closing.connect(osc_receiver.stop)
    osc_thread.finished.connect(osc_receiver.deleteLater)
    
    win.show()
    #win.setFocus()
    osc_thread.start()
    app.run()

