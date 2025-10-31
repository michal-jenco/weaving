# weaver.py
# PyQt5 weaving simulator - static textile-realistic
# Requirements: PyQt5, Pillow (Pillow only for optional save)
# Install: pip install PyQt5 Pillow

import sys
import math
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QPainter, QImage, QColor, QPen, QBrush, QPixmap, QLinearGradient
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton,
    QColorDialog, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox
)

CANVAS_SIZE = 1200  # fixed 1200x1200 output

def parse_input(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return []
    # ensure all lines same length — pad/truncate to first line length
    width = len(lines[0])
    parsed = []
    for ln in lines:
        if len(ln) < width:
            ln = ln.ljust(width, '0')
        elif len(ln) > width:
            ln = ln[:width]
        parsed.append([1 if c == '1' else 0 for c in ln])
    return parsed  # list of rows, top to bottom

def make_thread_gradient(color_rgb, vertical=True, thickness=1.0):
    """Return a QLinearGradient for a thread ribbon to give subtle cylindrical shading."""
    r, g, b = color_rgb
    grad = QLinearGradient()
    if vertical:
        grad.setStart(0, 0)
        grad.setFinalStop(0, 1)
    else:
        grad.setStart(0, 0)
        grad.setFinalStop(1, 0)
    # We use normalized stops (painter will map)
    # Slight highlight center and darker edges
    highlight = QColor(max(0, r + 40), max(0, g + 40), max(0, b + 40))
    darker = QColor(max(0, r - 40), max(0, g - 40), max(0, b - 40))
    grad.setColorAt(0.0, darker)
    grad.setColorAt(0.45, color_rgb_to_qcolor((r, g, b)))
    grad.setColorAt(0.5, highlight)
    grad.setColorAt(0.55, color_rgb_to_qcolor((r, g, b)))
    grad.setColorAt(1.0, darker)
    return grad

def color_rgb_to_qcolor(rgb):
    r, g, b = rgb
    return QColor(r, g, b)

class WeaverCanvas:
    """Render weaving into a QImage (static)."""

    def __init__(self, rows_cols, warp_rgb=(0,0,0), weft_rgb=(255,255,255), canvas_size=CANVAS_SIZE):
        self.grid = rows_cols  # list of rows (0/1)
        self.rows = len(rows_cols)
        self.cols = len(rows_cols[0]) if self.rows else 0
        self.warp_rgb = warp_rgb
        self.weft_rgb = weft_rgb
        self.size = canvas_size
        self.img = QImage(self.size, self.size, QImage.Format_RGB32)
        self.img.fill(QColor(240,240,240))  # neutral background
        self.painter = QPainter(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing)
        self.margin = int(self.size * 0.03)
        self.inner_size = self.size - 2 * self.margin

    def render(self):
        if self.rows == 0 or self.cols == 0:
            # empty canvas
            self.painter.fillRect(0, 0, self.size, self.size, QColor(240,240,240))
            self.painter.end()
            return self.img

        # compute thickness of threads based on grid
        # We'll make threads slightly thicker than grid cell size for overlap realism
        cell_h = self.inner_size / self.rows
        cell_w = self.inner_size / self.cols
        # thread thickness (ribbon width)
        warp_thickness = max(8, int(cell_w * 0.9))
        weft_thickness = max(8, int(cell_h * 0.9))

        # draw area background
        area_rect = QRectF(self.margin, self.margin, self.inner_size, self.inner_size)
        self.painter.fillRect(area_rect, QColor(220,220,220))

        # draw base weft (horizontal ribbons) across full area first
        for r in range(self.rows):
            # y center for this row's weft ribbon
            y = self.margin + r * cell_h + cell_h / 2.0
            self.draw_weft_ribbon(y, weft_thickness, cell_w, warp_thickness, r)

        # draw warp ribbons (vertical). For intersections where warp is under (0),
        # we'll draw small weft segments on top to simulate weft-over.
        for c in range(self.cols):
            x = self.margin + c * cell_w + cell_w / 2.0
            self.draw_warp_ribbon(x, warp_thickness, cell_h, weft_thickness, c)

        self.painter.end()
        return self.img

    def draw_weft_ribbon(self, center_y, thickness, cell_w, warp_thickness, row_index):
        """Draw a full horizontal ribbon (weft) across the fabric."""
        half = thickness / 2.0
        path_rect = QRectF(self.margin - warp_thickness, center_y - half,
                           self.inner_size + 2*warp_thickness, thickness)
        # create gradient horizontally across the ribbon to give curvature
        grad = QLinearGradient(path_rect.left(), path_rect.top(), path_rect.right(), path_rect.top())
        # subtle 3-stop: dark edges, highlight center
        base = color_rgb_to_qcolor(self.weft_rgb)
        dark = QColor(max(0, self.weft_rgb[0]-40), max(0, self.weft_rgb[1]-40), max(0, self.weft_rgb[2]-40))
        light = QColor(min(255, self.weft_rgb[0]+40), min(255, self.weft_rgb[1]+40), min(255, self.weft_rgb[2]+40))
        grad.setColorAt(0.0, dark)
        grad.setColorAt(0.5, light)
        grad.setColorAt(1.0, dark)
        self.painter.setBrush(QBrush(grad))
        self.painter.setPen(Qt.NoPen)

        # make ribbon wavy: draw many small slices with sinusoidal vertical offset
        slices = 120
        slice_w = (path_rect.width()) / slices
        amplitude = max(1.0, thickness * 0.12)
        freq = 2.0 * math.pi / max(120.0, slices/2.0)  # gentle waves
        x0 = path_rect.left()
        for i in range(slices):
            sx = x0 + i * slice_w
            # vertical offset for this slice
            offset = math.sin((i + row_index*3) * freq + row_index) * amplitude
            sr = QRectF(sx, path_rect.top() + offset, slice_w + 1, path_rect.height())
            self.painter.drawRect(sr)

        # add fiber lines: many thin slightly translucent short strokes along ribbon
        self.painter.setPen(QPen(QColor(0,0,0,24), 1))
        fiber_count = int(max(20, thickness * 0.7))
        for f in range(fiber_count):
            fx = path_rect.left() + random.random() * path_rect.width()
            fy = center_y + (random.random() - 0.5) * thickness * 0.6
            lx = fx + (random.random() - 0.5) * thickness * 0.7
            ly = fy + (random.random() - 0.5) * thickness * 0.7
            self.painter.drawLine(QPointF(fx, fy), QPointF(lx, ly))

    def draw_warp_ribbon(self, center_x, thickness, cell_h, weft_thickness, col_index):
        half = thickness / 2.0
        path_rect = QRectF(center_x - half, self.margin - weft_thickness,
                           thickness, self.inner_size + 2*weft_thickness)
        # gradient vertical
        grad = QLinearGradient(path_rect.left(), path_rect.top(), path_rect.left(), path_rect.bottom())
        base = color_rgb_to_qcolor(self.warp_rgb)
        dark = QColor(max(0, self.warp_rgb[0]-40), max(0, self.warp_rgb[1]-40), max(0, self.warp_rgb[2]-40))
        light = QColor(min(255, self.warp_rgb[0]+40), min(255, self.warp_rgb[1]+40), min(255, self.warp_rgb[2]+40))
        grad.setColorAt(0.0, dark)
        grad.setColorAt(0.5, light)
        grad.setColorAt(1.0, dark)
        self.painter.setBrush(QBrush(grad))
        self.painter.setPen(Qt.NoPen)

        # draw warp wavy slices
        slices = 120
        slice_h = (path_rect.height()) / slices
        amplitude = max(1.0, thickness * 0.12)
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        y0 = path_rect.top()
        for i in range(slices):
            sy = y0 + i * slice_h
            offset = math.sin((i + col_index*2) * freq + col_index) * amplitude
            sr = QRectF(path_rect.left() + offset, sy, path_rect.width(), slice_h + 1)
            self.painter.drawRect(sr)

        # fiber lines along warp
        self.painter.setPen(QPen(QColor(0,0,0,32), 1))
        fiber_count = int(max(20, thickness * 0.6))
        for f in range(fiber_count):
            fy = path_rect.top() + random.random() * path_rect.height()
            fx = center_x + (random.random() - 0.5) * thickness * 0.6
            lx = fx + (random.random() - 0.5) * thickness * 0.6
            ly = fy + (random.random() - 0.5) * thickness * 0.6
            self.painter.drawLine(QPointF(fx, fy), QPointF(lx, ly))

        # Now we need to simulate interlacing: for each row, check grid[row][col]
        # If warp is under (0), draw a short weft segment on top at that crossing to hide warp.
        # We'll draw a small horizontal rounded rect covering the crossing area.
        # compute cell height/width from grid sizes
        cell_h = self.inner_size / self.rows
        cell_w = self.inner_size / self.cols
        half_weft = weft_thickness / 2.0
        for r in range(self.rows):
            bit = self.grid[r][col_index]
            # center of crossing
            y = self.margin + r * cell_h + cell_h / 2.0
            x = center_x
            if bit == 0:
                # draw weft piece over warp here
                # small rise to match wavy weft shape: compute offset similar to draw_weft_ribbon
                # amplitude and freq same as used for weft
                slices = 120
                freq = 2.0 * math.pi / max(120.0, slices/2.0)
                # approximate offset by using column index to compute phase
                i = int((x - self.margin) / (self.inner_size / slices))
                offset = math.sin((i + r*3) * freq + r) * max(1.0, weft_thickness * 0.12)
                rect = QRectF(x - cell_w/2.0 - thickness/2.0, y - half_weft + offset,
                              cell_w + thickness, weft_thickness)
                # gradient for the small weft cap
                grad_local = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
                grad_local.setColorAt(0.0, QColor(max(0, self.weft_rgb[0]-30), max(0, self.weft_rgb[1]-30), max(0, self.weft_rgb[2]-30)))
                grad_local.setColorAt(0.5, QColor(min(255, self.weft_rgb[0]+30), min(255, self.weft_rgb[1]+30), min(255, self.weft_rgb[2]+30)))
                grad_local.setColorAt(1.0, QColor(max(0, self.weft_rgb[0]-30), max(0, self.weft_rgb[1]-30), max(0, self.weft_rgb[2]-30)))
                self.painter.setBrush(QBrush(grad_local))
                self.painter.setPen(Qt.NoPen)
                self.painter.drawRoundedRect(rect, 3, 3)

class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jacquard Punch-Card Weaver (Static, Textile-Realistic)")
        self.resize(1400, 900)

        # Widgets
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Enter binary punch card rows, one row per line. All lines should have same length.\nExample:\n01010101\n10101010\n01010101\n10101010")
        # default pattern: simple twill-ish
        self.input_edit.setPlainText("01010101\n10101010\n01010101\n10101010\n01010101\n10101010\n01010101\n10101010")

        self.warp_color_btn = QPushButton("Warp color (black)")
        self.weft_color_btn = QPushButton("Weft color (white)")
        self.weave_btn = QPushButton("Weave Pattern")
        self.save_btn = QPushButton("Save Image")

        self.image_label = QLabel()
        self.image_label.setFixedSize(CANVAS_SIZE//2, CANVAS_SIZE//2)  # preview scaled down
        self.image_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.image_label.setAlignment(Qt.AlignCenter)

        # colors
        self.warp_rgb = (0, 0, 0)
        self.weft_rgb = (255, 255, 255)

        # Layouts
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.input_edit)
        color_row = QHBoxLayout()
        color_row.addWidget(self.warp_color_btn)
        color_row.addWidget(self.weft_color_btn)
        left_layout.addLayout(color_row)
        left_layout.addWidget(self.weave_btn)
        left_layout.addWidget(self.save_btn)
        left_layout.addStretch()

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        right_layout.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)

        # Connections
        self.warp_color_btn.clicked.connect(self.pick_warp_color)
        self.weft_color_btn.clicked.connect(self.pick_weft_color)
        self.weave_btn.clicked.connect(self.on_weave)
        self.save_btn.clicked.connect(self.save_image)

        # initial render
        self.last_image = None
        self.on_weave()

    def pick_warp_color(self):
        col = QColorDialog.getColor(QColor(*self.warp_rgb), self, "Choose warp color")
        if col.isValid():
            self.warp_rgb = (col.red(), col.green(), col.blue())
            self.warp_color_btn.setText(f"Warp color ({col.name()})")
            self.on_weave()

    def pick_weft_color(self):
        col = QColorDialog.getColor(QColor(*self.weft_rgb), self, "Choose weft color")
        if col.isValid():
            self.weft_rgb = (col.red(), col.green(), col.blue())
            self.weft_color_btn.setText(f"Weft color ({col.name()})")
            self.on_weave()

    def on_weave(self):
        text = self.input_edit.toPlainText()
        grid = parse_input(text)
        if not grid:
            QMessageBox.warning(self, "No data", "Please enter at least one line of binary input.")
            return
        # ensure rectangular grid
        rows = len(grid)
        cols = len(grid[0])
        # render
        renderer = WeaverCanvas(grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb, canvas_size=CANVAS_SIZE)
        qimg = renderer.render()
        self.last_image = qimg
        # convert to pixmap scaled for preview
        preview = qimg.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(preview))

    def save_image(self):
        if self.last_image is None:
            QMessageBox.information(self, "No image", "Nothing to save. Press 'Weave Pattern' first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save woven image", "weave.png", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
        if path:
            # QImage save
            saved = self.last_image.save(path)
            if saved:
                QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")
            else:
                QMessageBox.warning(self, "Save failed", "Could not save image.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WeavingApp()
    window.show()
    sys.exit(app.exec_())
