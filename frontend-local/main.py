import sys
import cv2
import numpy as np
import torch
from skimage.transform import resize
from transformers import AutoProcessor, AutoModelForMaskGeneration
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QTabWidget, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout
from process import postProcess
import firebase_admin
from time import time
from firebase_admin import credentials, firestore
from base64 import b64encode


class CameraThread(QThread):
    image_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.image_signal.emit(frame)

    def stop(self):
        self.capture.release()


class ProcessImagesThread(QThread):
    result_signal = pyqtSignal(QImage, int, int)

    def __init__(self, processor, model, captured_images):
        super().__init__()
        self.processor = processor
        self.model = model
        self.captured_images = captured_images

    def run(self):
        for index, frame in enumerate(self.captured_images):
            frame_resized = resize(
                frame, (256, 256), mode='constant', preserve_range=True)
            image = cv2.cvtColor(frame_resized.astype(
                np.uint8), cv2.COLOR_RGB2BGR)

            inputs = self.processor(image, return_tensors="pt").to("cpu")

            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            frame_with_mask_rgb, nuclei_count, adj_nuclei_count = postProcess(
                frame, outputs)

            image_qt = QImage(
                frame_with_mask_rgb, frame_with_mask_rgb.shape[1], frame_with_mask_rgb.shape[0], QImage.Format_RGB888)
            self.result_signal.emit(image_qt, nuclei_count, adj_nuclei_count)
        self.captured_images.clear()


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Project NS")
        self.setGeometry(100, 100, 450, 200)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")
        self.model = AutoModelForMaskGeneration.from_pretrained(
            "facebook/sam-vit-base")

        cred = credentials.Certificate("./serviceAccountKey.json")

        app = firebase_admin.initialize_app(cred)
        self.db = firestore.client()

        try:
            self.model.load_state_dict(torch.load(
                'SAM_finetuned.pth', weights_only=True, map_location=torch.device("cpu")))
            print("[INFO] Loaded custom weights for SAM model.")
        except Exception as e:
            print(f"[WARNING] Could not load custom weights: {e}")

        self.model.to("cpu")
        self.model.eval()

        self.captured_images_tab = QWidget()
        self.result_tab = QWidget()
        self.tabs.addTab(self.captured_images_tab, "Captured Images")
        self.tabs.addTab(self.result_tab, "Results")

        self.init_captured_images_tab()
        self.init_result_tab()

        self.camera_thread = CameraThread()
        self.camera_thread.image_signal.connect(self.update_image)
        self.camera_thread.start()

        self.captured_images = []
        self.processed_images = []

    def init_captured_images_tab(self):
        layout = QVBoxLayout()

        layout.addStretch(1)

        self.label = QLabel()
        layout.addWidget(self.label)

        layout.addSpacing(20)

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button)

        layout.addSpacing(10)

        self.process_button = QPushButton("Process Images")
        self.process_button.clicked.connect(self.process_images)
        layout.addWidget(self.process_button)

        layout.addSpacing(20)

        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)

        self.image_layout = QHBoxLayout()
        self.image_container = QWidget()
        self.image_container.setLayout(self.image_layout)
        self.image_scroll_area.setWidget(self.image_container)

        layout.addWidget(self.image_scroll_area)

        layout.addStretch(1)

        self.captured_images_tab.setLayout(layout)

    def init_result_tab(self):
        layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.result_image_container = QWidget()
        self.result_image_layout = QVBoxLayout()
        self.result_image_container.setLayout(self.result_image_layout)

        self.scroll_area.setWidget(self.result_image_container)
        layout.addWidget(self.scroll_area)

        self.result_tab.setLayout(layout)

    def update_image(self, frame):
        frame = cv2.resize(frame, (256, 256))
        image = QImage(frame, frame.shape[1],
                       frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(image))

    def capture_image(self):
        ret, frame = self.camera_thread.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.captured_images.append(frame)
            frame_resized = cv2.resize(frame, (800, 800))

            image = QImage(
                frame_resized, frame_resized.shape[1], frame_resized.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)

            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
            self.image_layout.addWidget(image_label)

            self.image_scroll_area.horizontalScrollBar().setValue(
                self.image_scroll_area.horizontalScrollBar().maximum())

    def process_images(self):
        self.process_images_thread = ProcessImagesThread(
            self.processor, self.model, self.captured_images)
        self.process_images_thread.result_signal.connect(self.display_results)
        self.process_images_thread.start()

    def QImageToCvMat(self, incomingImage, target_size=(128, 128)):
        if incomingImage.isNull():
            return np.array([])

        width = incomingImage.width()
        height = incomingImage.height()
        fmt = incomingImage.format()

        if fmt == QImage.Format_RGB32:
            ptr = incomingImage.bits()
            ptr.setsize(incomingImage.byteCount())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                (height, width, 4))
            cv_mat = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        elif fmt == QImage.Format_RGB888:
            ptr = incomingImage.bits()
            ptr.setsize(incomingImage.byteCount())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                (height, width, 3))
            cv_mat = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        elif fmt == QImage.Format_Grayscale8:
            ptr = incomingImage.bits()
            ptr.setsize(incomingImage.byteCount())
            cv_mat = np.frombuffer(
                ptr, dtype=np.uint8).reshape((height, width))

        else:
            incomingImage = incomingImage.convertToFormat(QImage.Format_RGB888)
            return QImageToCvMat(incomingImage, target_size)

        resized_cv_mat = cv2.resize(
            cv_mat, target_size, interpolation=cv2.INTER_AREA)

        return resized_cv_mat

    def display_results(self, image_qt, nuclei_count, adj_nuclei_count):
        result_label = QLabel()
        result_label.setPixmap(QPixmap.fromImage(
            image_qt).scaled(256, 256, Qt.KeepAspectRatio))

        count_label = QLabel(
            f"Nuclei Count: {nuclei_count}, Adjusted Nuclei Count: {adj_nuclei_count}")

        self.result_image_layout.addWidget(result_label)
        self.result_image_layout.addWidget(count_label)

        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum())
        frame = image_qt
        _, jpeg = cv2.imencode('.png', self.QImageToCvMat(frame))

        im_b64 = b64encode(jpeg.tobytes()).decode()

        self.db.collection('Images').add({
            # 'original_image': b64encode(og_jpeg.tobytes()).decode(),
            'segmented_image': im_b64,
            'nuclei_count': nuclei_count,
            'adjusted_nuclei_count': adj_nuclei_count,
            'time': time()
        })


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec_())
