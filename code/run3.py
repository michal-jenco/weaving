# weaver_enhanced.py
# Enhanced PyQt5 weaving simulator with pattern generator, glitch controls, presets,
# image import, animation, seed/rewind, and thumbnail history.
# Requirements: PyQt5, Pillow
# Install: pip install PyQt5 Pillow

import sys
import math
import random
from PIL import Image
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, QSize
from PyQt5.QtGui import QPainter, QImage, QColor, QPen, QBrush, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QColorDialog,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QSpinBox, QSlider,
    QComboBox, QCheckBox, QListWidget, QListWidgetItem, QLineEdit
)

CANVAS_SIZE = 1200

# -------------------- utilities --------------------

def color_rgb_to_qcolor(rgb):
    r, g, b = rgb
    return QColor(r, g, b)


def parse_input(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return []
    width = len(lines[0])
    parsed = []
    for ln in lines:
        if len(ln) < width:
            ln = ln.ljust(width, '0')
        elif len(ln) > width:
            ln = ln[:width]
        parsed.append([1 if c == '1' else 0 for c in ln])
    return parsed

# -------------------- pattern generators --------------------

def gen_checkerboard(rows, cols, seed=None):
    if seed is not None:
        random.seed(seed)
    grid = [[(r + c) % 2 for c in range(cols)] for r in range(rows)]
    return grid


def gen_stripes(rows, cols, vertical=True, period=2):
    if vertical:
        grid = [[(c // period) % 2 for c in range(cols)] for r in range(rows)]
    else:
        grid = [[(r // period) % 2 for c in range(cols)] for r in range(rows)]
    return grid


def gen_diagonal(rows, cols):
    grid = [[((r + c) % 4 < 2) * 1 for c in range(cols)] for r in range(rows)]
    return grid


def gen_random(rows, cols, randomness=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    grid = [[1 if random.random() > randomness else 0 for c in range(cols)] for r in range(rows)]
    return grid


def gen_triangles(rows, cols):
    grid = [[1 if c < (cols * (r / rows)) else 0 for c in range(cols)] for r in range(rows)]
    return grid


def gen_concentric(rows, cols, rings=5):
    cx = cols / 2.0
    cy = rows / 2.0
    maxd = math.hypot(cx, cy)
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            d = math.hypot(c - cx, r - cy)
            ring = int((d / maxd) * rings)
            row.append(ring % 2)
        grid.append(row)
    return grid


def mirror_symmetry(grid, axis='vertical'):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    out = [[0]*cols for _ in range(rows)]
    if axis == 'vertical':
        for r in range(rows):
            for c in range(cols):
                out[r][c] = grid[r][c] if c < cols//2 else grid[r][cols - 1 - c]
    else:
        for r in range(rows):
            for c in range(cols):
                out[r][c] = grid[r][c] if r < rows//2 else grid[rows - 1 - r][c]
    return out

# -------------------- glitch application --------------------

def apply_glitch(grid, glitch_pct=0.0, smear_pct=0.0, seed=None):
    if seed is not None:
        random.seed(seed)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    out = [list(row) for row in grid]
    flips = int(round(glitch_pct * rows * cols))
    # uniform random flips
    for _ in range(flips):
        r = random.randrange(rows)
        c = random.randrange(cols)
        out[r][c] = 1 - out[r][c]
    # smear: shift some rows by -1/0/+1 with probability smear_pct
    if smear_pct > 0:
        for r in range(rows):
            if random.random() < smear_pct:
                shift = random.choice([-2, -1, 0, 1, 2])
                out[r] = out[r][-shift:] + out[r][:-shift] if shift != 0 else out[r]
    return out

# -------------------- image import --------------------


def image_to_binary_grid(path, rows, cols, threshold=128, invert=False):
    img = Image.open(path).convert('L')
    img = img.resize((cols, rows), Image.LANCZOS)
    px = img.load()
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = px[c, r]
            bit = 1 if v < threshold else 0
            if invert:
                bit = 1 - bit
            row.append(bit)
        grid.append(row)
    return grid

# -------------------- Renderer (based on your original) --------------------

class WeaverCanvas:
    def __init__(self, rows_cols, warp_rgb=(0,0,0), weft_rgb=(255,255,255), canvas_size=CANVAS_SIZE, tension=1.0):
        self.grid = rows_cols
        self.rows = len(rows_cols)
        self.cols = len(rows_cols[0]) if self.rows else 0
        self.warp_rgb = warp_rgb
        self.weft_rgb = weft_rgb
        self.size = canvas_size
        self.img = QImage(self.size, self.size, QImage.Format_RGB32)
        self.img.fill(QColor(240,240,240))
        self.painter = QPainter(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing)
        self.margin = int(self.size * 0.03)
        self.inner_size = self.size - 2 * self.margin
        self.tension = tension

    def render(self, partial_progress=None):
        # partial_progress: None or (rows_drawn) to animate progressive weave
        if self.rows == 0 or self.cols == 0:
            self.painter.fillRect(0, 0, self.size, self.size, QColor(240,240,240))
            self.painter.end()
            return self.img

        cell_h = self.inner_size / self.rows
        cell_w = self.inner_size / self.cols
        warp_thickness = max(6, int(cell_w * 0.9 * self.tension))
        weft_thickness = max(6, int(cell_h * 0.9 * self.tension))

        area_rect = QRectF(self.margin, self.margin, self.inner_size, self.inner_size)
        self.painter.fillRect(area_rect, QColor(220,220,220))

        # draw weft rows; if partial_progress is set, only draw up to that many rows
        rows_to_draw = self.rows if partial_progress is None else min(self.rows, partial_progress)
        for r in range(rows_to_draw):
            y = self.margin + r * cell_h + cell_h / 2.0
            self.draw_weft_ribbon(y, weft_thickness, cell_w, warp_thickness, r)

        # draw warp ribbons fully (so warp looks like vertical threads with correct overlap)
        for c in range(self.cols):
            x = self.margin + c * cell_w + cell_w / 2.0
            self.draw_warp_ribbon(x, warp_thickness, cell_h, weft_thickness, c)

        self.painter.end()
        return self.img

    def draw_weft_ribbon(self, center_y, thickness, cell_w, warp_thickness, row_index):
        half = thickness / 2.0
        path_rect = QRectF(self.margin - warp_thickness, center_y - half,
                           self.inner_size + 2*warp_thickness, thickness)
        grad = QPixmap(1,1)  # placeholder not used; we'll draw slices with base color
        base = color_rgb_to_qcolor(self.weft_rgb)
        dark = QColor(max(0, self.weft_rgb[0]-40), max(0, self.weft_rgb[1]-40), max(0, self.weft_rgb[2]-40))
        light = QColor(min(255, self.weft_rgb[0]+40), min(255, self.weft_rgb[1]+40), min(255, self.weft_rgb[2]+40))
        self.painter.setPen(Qt.NoPen)

        slices = 120
        slice_w = (path_rect.width()) / slices
        amplitude = max(1.0, thickness * 0.12) * (1.0 / max(0.5, self.tension))
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        x0 = path_rect.left()
        for i in range(slices):
            sx = x0 + i * slice_w
            offset = math.sin((i + row_index*3) * freq + row_index) * amplitude
            # gradient simulated by alternating brush alpha
            col = QColor(self.weft_rgb[0], self.weft_rgb[1], self.weft_rgb[2])
            if (i % 8) < 4:
                col = QColor(min(255, col.red()+10), min(255, col.green()+10), min(255, col.blue()+10), 255)
            else:
                col = QColor(max(0, col.red()-10), max(0, col.green()-10), max(0, col.blue()-10), 255)
            self.painter.setBrush(QBrush(col))
            sr = QRectF(sx, path_rect.top() + offset, slice_w + 1, path_rect.height())
            self.painter.drawRect(sr)

        # fiber lines
        self.painter.setPen(QPen(QColor(0,0,0,24), 1))
        fiber_count = int(max(12, thickness * 0.5))
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
        self.painter.setPen(Qt.NoPen)
        slices = 120
        slice_h = (path_rect.height()) / slices
        amplitude = max(1.0, thickness * 0.12) * (1.0 / max(0.5, self.tension))
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        y0 = path_rect.top()
        for i in range(slices):
            sy = y0 + i * slice_h
            offset = math.sin((i + col_index*2) * freq + col_index) * amplitude
            col = QColor(self.warp_rgb[0], self.warp_rgb[1], self.warp_rgb[2])
            if (i % 8) < 4:
                col = QColor(min(255, col.red()+10), min(255, col.green()+10), min(255, col.blue()+10), 255)
            else:
                col = QColor(max(0, col.red()-10), max(0, col.green()-10), max(0, col.blue()-10), 255)
            self.painter.setBrush(QBrush(col))
            sr = QRectF(path_rect.left() + offset, sy, path_rect.width(), slice_h + 1)
            self.painter.drawRect(sr)

        self.painter.setPen(QPen(QColor(0,0,0,32), 1))
        fiber_count = int(max(12, thickness * 0.5))
        for f in range(fiber_count):
            fy = path_rect.top() + random.random() * path_rect.height()
            fx = center_x + (random.random() - 0.5) * thickness * 0.6
            lx = fx + (random.random() - 0.5) * thickness * 0.6
            ly = fy + (random.random() - 0.5) * thickness * 0.6
            self.painter.drawLine(QPointF(fx, fy), QPointF(lx, ly))

        # weft caps where warp is under
        cell_h = self.inner_size / self.rows
        cell_w = self.inner_size / self.cols
        half_weft = weft_thickness / 2.0
        slices = 120
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        for r in range(self.rows):
            bit = self.grid[r][col_index]
            y = self.margin + r * cell_h + cell_h / 2.0
            x = center_x
            if bit == 0:
                i = int((x - self.margin) / (self.inner_size / slices))
                offset = math.sin((i + r*3) * freq + r) * max(1.0, weft_thickness * 0.12)
                rect = QRectF(x - cell_w/2.0 - thickness/2.0, y - half_weft + offset,
                              cell_w + thickness, weft_thickness)
                col = QColor(self.weft_rgb[0], self.weft_rgb[1], self.weft_rgb[2])
                grad_local_left = QColor(max(0, self.weft_rgb[0]-30), max(0, self.weft_rgb[1]-30), max(0, self.weft_rgb[2]-30))
                grad_local_mid = QColor(min(255, self.weft_rgb[0]+30), min(255, self.weft_rgb[1]+30), min(255, self.weft_rgb[2]+30))
                # draw rounded rect
                self.painter.setBrush(QBrush(grad_local_mid))
                self.painter.setPen(Qt.NoPen)
                self.painter.drawRoundedRect(rect, 3, 3)

# -------------------- Main App --------------------

class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jacquard Weaver — Enhanced")
        self.resize(1500, 960)

        # core widgets
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Enter binary rows or use the generator/import tools")
        self.input_edit.setPlainText("01010101\n10101010\n01010101\n10101010\n01010101\n10101010\n01010101\n10101010")

        self.warp_color_btn = QPushButton("Warp color (black)")
        self.weft_color_btn = QPushButton("Weft color (white)")
        self.weave_btn = QPushButton("Weave Pattern")
        self.save_btn = QPushButton("Save Image")
        self.import_btn = QPushButton("Import image -> binary")

        self.image_label = QLabel()
        self.image_label.setFixedSize(CANVAS_SIZE//2, CANVAS_SIZE//2)
        self.image_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        self.image_label.setAlignment(Qt.AlignCenter)

        # generator controls
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(4, 256); self.rows_spin.setValue(32)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(4, 256); self.cols_spin.setValue(32)
        self.pattern_combo = QComboBox();
        self.pattern_combo.addItems(['Checkerboard','Stripes Vert','Stripes Horz','Diagonal','Triangles','Concentric','Random'])
        self.randomness_slider = QSlider(Qt.Horizontal); self.randomness_slider.setRange(0,100); self.randomness_slider.setValue(50)
        self.gen_btn = QPushButton('Generate')
        self.seed_edit = QLineEdit(); self.seed_edit.setPlaceholderText('seed (optional)')

        # glitch controls
        self.glitch_slider = QSlider(Qt.Horizontal); self.glitch_slider.setRange(0,100); self.glitch_slider.setValue(0)
        self.smear_slider = QSlider(Qt.Horizontal); self.smear_slider.setRange(0,100); self.smear_slider.setValue(0)
        self.glitch_anim_chk = QCheckBox('Temporal glitch (animate)')
        self.anim_speed_slider = QSlider(Qt.Horizontal); self.anim_speed_slider.setRange(10,1000); self.anim_speed_slider.setValue(120)

        # tension/depth
        self.tension_slider = QSlider(Qt.Horizontal); self.tension_slider.setRange(5,200); self.tension_slider.setValue(100)

        # history thumbnails
        self.history_list = QListWidget(); self.history_list.setIconSize(QSize(160,160))
        self.clear_history_btn = QPushButton('Clear History')

        # layouts
        left = QVBoxLayout()
        left.addWidget(QLabel('Binary Input / Generated Pattern'))
        left.addWidget(self.input_edit)

        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel('Rows')); gen_row.addWidget(self.rows_spin)
        gen_row.addWidget(QLabel('Cols')); gen_row.addWidget(self.cols_spin)
        left.addLayout(gen_row)

        gen2 = QHBoxLayout(); gen2.addWidget(QLabel('Pattern')); gen2.addWidget(self.pattern_combo); gen2.addWidget(QLabel('Randomness')); gen2.addWidget(self.randomness_slider)
        left.addLayout(gen2)
        gen3 = QHBoxLayout(); gen3.addWidget(QLabel('Seed')); gen3.addWidget(self.seed_edit); gen3.addWidget(self.gen_btn); gen3.addWidget(self.import_btn)
        left.addLayout(gen3)

        left.addWidget(QLabel('Glitch % (flip bits)'))
        left.addWidget(self.glitch_slider)
        left.addWidget(QLabel('Smear % (row shifts)'))
        left.addWidget(self.smear_slider)
        left.addWidget(self.glitch_anim_chk)
        left.addWidget(QLabel('Temporal speed (ms per step)'))
        left.addWidget(self.anim_speed_slider)

        left.addWidget(QLabel('Thread tension (visual)'))
        left.addWidget(self.tension_slider)

        left.addLayout(self._make_color_row())
        left.addWidget(self.weave_btn)
        left.addWidget(self.save_btn)
        left.addStretch()

        right = QVBoxLayout()
        right.addWidget(self.image_label, alignment=Qt.AlignCenter)
        right.addWidget(QLabel('History (click to preview)'))
        right.addWidget(self.history_list)
        hrow = QHBoxLayout(); hrow.addWidget(self.clear_history_btn); right.addLayout(hrow)
        right.addStretch()

        main = QHBoxLayout(); main.addLayout(left, stretch=3); main.addLayout(right, stretch=2)
        self.setLayout(main)

        # state
        self.warp_rgb = (0,0,0); self.weft_rgb = (255,255,255)
        self.last_image = None
        self.history = []

        # connections
        self.warp_color_btn.clicked.connect(self.pick_warp_color)
        self.weft_color_btn.clicked.connect(self.pick_weft_color)
        self.weave_btn.clicked.connect(self.on_weave)
        self.save_btn.clicked.connect(self.save_image)
        self.gen_btn.clicked.connect(self.on_generate)
        self.import_btn.clicked.connect(self.on_import)
        self.clear_history_btn.clicked.connect(self.on_clear_history)
        self.history_list.itemClicked.connect(self.on_history_click)

        # animation timer for temporal glitch / progressive weaving
        self.timer = QTimer()
        self.timer.timeout.connect(self._anim_step)
        self.anim_progress = 0

        # initial render
        self.on_weave()

    def _make_color_row(self):
        row = QHBoxLayout()
        row.addWidget(self.warp_color_btn); row.addWidget(self.weft_color_btn)
        return row

    # ---------------- UI actions ----------------
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

    def on_generate(self):
        rows = self.rows_spin.value(); cols = self.cols_spin.value()
        seed_text = self.seed_edit.text().strip()
        seed = None if seed_text == '' else int(seed_text) if seed_text.isdigit() else None
        patt = self.pattern_combo.currentText()
        rnd = self.randomness_slider.value() / 100.0
        if patt == 'Checkerboard':
            grid = gen_checkerboard(rows, cols, seed)
        elif patt == 'Stripes Vert':
            grid = gen_stripes(rows, cols, vertical=True, period=max(1,int(1 + rnd*6)))
        elif patt == 'Stripes Horz':
            grid = gen_stripes(rows, cols, vertical=False, period=max(1,int(1 + rnd*6)))
        elif patt == 'Diagonal':
            grid = gen_diagonal(rows, cols)
        elif patt == 'Triangles':
            grid = gen_triangles(rows, cols)
        elif patt == 'Concentric':
            rings = max(2, int(2 + rnd*10))
            grid = gen_concentric(rows, cols, rings=rings)
        else:
            grid = gen_random(rows, cols, randomness=1.0-rnd, seed=seed)
        # apply symmetry optionally using randomness toggles (mirror for readability)
        if rnd > 0.8:
            grid = mirror_symmetry(grid, axis='vertical')
        # put into input_edit as text
        txt = '\n'.join(''.join('1' if c else '0' for c in row) for row in grid)
        self.input_edit.setPlainText(txt)
        self.on_weave()

    def on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open image', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif)')
        if not path:
            return
        rows = self.rows_spin.value(); cols = self.cols_spin.value()
        try:
            grid = image_to_binary_grid(path, rows, cols, threshold=128, invert=False)
        except Exception as e:
            QMessageBox.warning(self, 'Import failed', f'Could not import image:\n{e}')
            return
        txt = '\n'.join(''.join('1' if c else '0' for c in row) for row in grid)
        self.input_edit.setPlainText(txt)
        self.on_weave()

    def on_clear_history(self):
        self.history = []
        self.history_list.clear()

    def on_history_click(self, item):
        idx = item.data(Qt.UserRole)
        if idx is None: return
        img = self.history[idx]
        preview = img.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(preview))

    # ---------------- rendering pipeline ----------------
    def _prepare_grid_from_input(self):
        text = self.input_edit.toPlainText()
        grid = parse_input(text)
        if not grid:
            return None
        # apply glitch transformations
        glitch_pct = self.glitch_slider.value() / 100.0
        smear_pct = self.smear_slider.value() / 100.0
        seed_text = self.seed_edit.text().strip()
        seed = None if seed_text == '' else int(seed_text) if seed_text.isdigit() else None
        grid = apply_glitch(grid, glitch_pct=glitch_pct, smear_pct=smear_pct, seed=seed)
        return grid

    def on_weave(self):
        grid = self._prepare_grid_from_input()
        if grid is None:
            QMessageBox.warning(self, 'No data', 'Please enter or generate a binary grid first.')
            return
        tension = self.tension_slider.value() / 100.0
        renderer = WeaverCanvas(grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb, canvas_size=CANVAS_SIZE, tension=tension)

        # temporal glitch animation: animate flips over time instead of static application
        if self.glitch_anim_chk.isChecked():
            self.anim_grid_base = parse_input(self.input_edit.toPlainText())
            self.anim_grid = [list(r) for r in self.anim_grid_base]
            self.anim_progress = 0
            interval = max(10, self.anim_speed_slider.value())
            self.timer.start(interval)
            return

        qimg = renderer.render()
        self._set_last_image_and_history(qimg)

    def _anim_step(self):
        # progressively apply random flips and re-render a partial progressive weave
        if not hasattr(self, 'anim_grid'):
            self.timer.stop(); return
        rows = len(self.anim_grid)
        cols = len(self.anim_grid[0]) if rows else 0
        # step: flip a handful of bits per step
        flips = max(1, int(rows*cols*0.005))
        for _ in range(flips):
            r = random.randrange(rows)
            c = random.randrange(cols)
            self.anim_grid[r][c] = 1 - self.anim_grid[r][c]
        # progressive render: increase rows drawn
        self.anim_progress = min(rows, self.anim_progress + max(1, rows//20))
        renderer = WeaverCanvas(self.anim_grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb, canvas_size=CANVAS_SIZE, tension=self.tension_slider.value()/100.0)
        qimg = renderer.render(partial_progress=self.anim_progress)
        self._set_last_image_and_history(qimg, add_history_preview=False)
        # if full progress reached, stop and add to history
        if self.anim_progress >= rows:
            self.timer.stop()
            self._set_last_image_and_history(qimg)

    def _set_last_image_and_history(self, qimg, add_history_preview=True):
        self.last_image = qimg
        preview = qimg.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(preview))
        if add_history_preview:
            # keep up to 12 thumbnails
            if len(self.history) >= 12:
                self.history.pop(0)
                self.history_list.takeItem(0)
            self.history.append(qimg.copy())
            item = QListWidgetItem()
            icon = QPixmap.fromImage(qimg).scaled(160,160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(QIcon(icon))
            item.setData(Qt.UserRole, len(self.history)-1)
            self.history_list.addItem(item)

    def save_image(self):
        if self.last_image is None:
            QMessageBox.information(self, 'No image', "Nothing to save. Press 'Weave Pattern' first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save woven image', 'weave.png', 'PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)')
        if path:
            saved = self.last_image.save(path)
            if saved:
                QMessageBox.information(self, 'Saved', f'Image saved to:\n{path}')
            else:
                QMessageBox.warning(self, 'Save failed', 'Could not save image.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = WeavingApp()
    w.show()
    sys.exit(app.exec_())
