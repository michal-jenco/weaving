# weaver_enhanced_final.py
# Full-featured Jacquard loom simulator (PyQt5 + Pillow)
# Requirements: PyQt5, Pillow
# pip install PyQt5 Pillow

import sys
import math
import random
from PIL import Image, ImageOps
from PyQt5.QtCore import Qt, QTimer, QSize, QRectF
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter, QPen, QBrush, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QColorDialog, QSlider, QComboBox, QFileDialog, QMessageBox, QSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QLineEdit, QGroupBox, QRadioButton
)

CANVAS_SIZE = 1200  # high-res render size for saving/export

# -------------------- Utilities --------------------

def clamp(v, a, b): return max(a, min(b, v))

def color_tuple_to_qcolor(rgb):
    return QColor(rgb[0], rgb[1], rgb[2])

def qimage_to_pil(qimg: QImage) -> Image.Image:
    buffer = qimg.bits().asstring(qimg.byteCount())
    pil = Image.frombuffer("RGBA", (qimg.width(), qimg.height()), buffer, "raw", "BGRA", 0, 1)
    return pil.convert("RGBA")

def pil_to_qimage(pil_img: Image.Image) -> QImage:
    rgba = pil_img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.width, rgba.height, QImage.Format_RGBA8888)
    return qimg

# -------------------- Parsing / IO --------------------

def parse_input(text):
    """Convert text lines of 0/1 into grid (list of rows)."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return []
    width = len(lines[0])
    parsed = []
    for ln in lines:
        ln2 = ln.ljust(width, '0')[:width]
        parsed.append([1 if c == '1' or c == '█' else 0 for c in ln2])
    return parsed

def grid_to_text(grid):
    return "\n".join("".join('1' if c else '0' for c in row) for row in grid)

def save_pattern_to_file(grid, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(grid_to_text(grid))

def load_pattern_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    return parse_input(txt), txt

# -------------------- Pattern Generators --------------------

def gen_checkerboard(rows, cols, seed=None):
    rnd = random.Random(seed)
    return [[(r + c) % 2 for c in range(cols)] for r in range(rows)]

def gen_stripes(rows, cols, vertical=True, period=2, seed=None):
    rnd = random.Random(seed)
    if vertical:
        return [[(c // period) % 2 for c in range(cols)] for r in range(rows)]
    else:
        return [[(r // period) % 2 for c in range(cols)] for r in range(rows)]

def gen_diagonal(rows, cols, seed=None):
    return [[1 if ((r + c) % 4) < 2 else 0 for c in range(cols)] for r in range(rows)]

def gen_triangles(rows, cols, seed=None):
    return [[1 if c < (cols * (r / max(1, rows))) else 0 for c in range(cols)] for r in range(rows)]

def gen_concentric(rows, cols, rings=6, seed=None):
    cx = (cols - 1) / 2.0
    cy = (rows - 1) / 2.0
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

def gen_random(rows, cols, randomness=0.5, seed=None):
    rnd = random.Random(seed)
    return [[1 if rnd.random() > randomness else 0 for _ in range(cols)] for _ in range(rows)]

def mirror_symmetry(grid, axis='vertical'):
    rows = len(grid); cols = len(grid[0]) if rows else 0
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

# -------------------- Glitch / Transform --------------------

def apply_glitch_static(grid, flip_pct=0.0, smear_pct=0.0, seed=None):
    rnd = random.Random(seed)
    rows = len(grid); cols = len(grid[0]) if rows else 0
    out = [list(row) for row in grid]
    # flips: each cell independently flips with probability flip_pct
    for r in range(rows):
        for c in range(cols):
            if rnd.random() < flip_pct:
                out[r][c] = 1 - out[r][c]
    # smear: shift whole row by small offset with probability smear_pct
    for r in range(rows):
        if rnd.random() < smear_pct:
            shift = rnd.choice([-3, -2, -1, 0, 1, 2, 3])
            if shift:
                out[r] = out[r][-shift:] + out[r][:-shift]
    return out

# -------------------- Image Import --------------------

def image_to_binary_grid(path, rows, cols, threshold=128, invert=False):
    pil = Image.open(path).convert('L')
    pil = ImageOps.autocontrast(pil)
    pil = pil.resize((cols, rows), Image.LANCZOS)
    px = pil.load()
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

# -------------------- Weaver Renderer --------------------

class WeaverCanvas:
    """
    A renderer that draws textile-realistic ribbons for warp and weft.
    Supports three visual styles: realistic, flat, high-contrast glitch overlay.
    """

    def __init__(self, grid, warp_rgb=(0,0,0), weft_rgb=(255,255,255), canvas_size=CANVAS_SIZE, tension=1.0, visual_style='realistic'):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.warp_rgb = warp_rgb
        self.weft_rgb = weft_rgb
        self.size = canvas_size
        self.tension = clamp(tension, 0.2, 3.0)
        self.visual_style = visual_style  # 'realistic' | 'flat' | 'high-contrast'
        self.img = QImage(self.size, self.size, QImage.Format_RGB32)
        self.img.fill(QColor(240,240,240))

    def render(self, partial_progress=None, jitter_hue=0.0):
        p = QPainter(self.img)
        p.setRenderHint(QPainter.Antialiasing)
        margin = int(self.size * 0.03)
        inner = self.size - 2*margin
        if self.rows == 0 or self.cols == 0:
            p.fillRect(0,0,self.size,self.size, QColor(240,240,240))
            p.end()
            return self.img

        cell_h = inner / self.rows
        cell_w = inner / self.cols

        # thickness
        warp_thickness = max(3, int(cell_w * 0.9 * self.tension))
        weft_thickness = max(3, int(cell_h * 0.9 * self.tension))

        # background
        p.fillRect(margin, margin, inner, inner, QColor(220,220,220))

        # flat style uses solid pens; realistic uses wavy ribbons with slices
        if self.visual_style == 'flat':
            # draw full weft rows
            pen_weft = QPen(QColor(*self.weft_rgb))
            pen_weft.setWidthF(weft_thickness)
            pen_warp = QPen(QColor(*self.warp_rgb))
            pen_warp.setWidthF(warp_thickness)
            p.setPen(pen_weft)
            for r in range(self.rows):
                y = margin + (r+0.5)*cell_h
                p.drawLine(margin, y, margin+inner, y)
            p.setPen(pen_warp)
            for c in range(self.cols):
                x = margin + (c+0.5)*cell_w
                p.drawLine(x, margin, x, margin+inner)
        else:
            # realistic or high-contrast: draw ribbons as many slices producing slight wave + fibers
            slices = 120
            # draw weft ribbons (horizontal)
            for r in range(self.rows if partial_progress is None else min(self.rows, partial_progress)):
                center_y = margin + r*cell_h + cell_h/2.0
                self._draw_weft_ribbon(p, center_y, weft_thickness, cell_w, warp_thickness, r, inner, slices)
            # draw warp ribbons (vertical)
            for c in range(self.cols):
                center_x = margin + c*cell_w + cell_w/2.0
                self._draw_warp_ribbon(p, center_x, warp_thickness, cell_h, weft_thickness, c, inner, slices)

        # interlacing caps: small weft caps where warp is under (0)
        cap_w = max(2, cell_w * 0.7)
        cap_h = max(2, cell_h * 0.6)
        p.setPen(Qt.NoPen)
        for r in range(self.rows):
            for c in range(self.cols):
                bit = self.grid[r][c]
                if bit == 0:
                    cx = margin + (c+0.5)*cell_w
                    cy = margin + (r+0.5)*cell_h
                    col = QColor(*self.weft_rgb)
                    p.setBrush(QBrush(col))
                    p.drawRoundedRect(QRectF(cx - cap_w/2, cy - cap_h/2, cap_w, cap_h), 2.0, 2.0)

        # high-contrast glitch overlay (subtle)
        if self.visual_style == 'high-contrast':
            # add hue jitter stripes overlay
            overlay = QImage(self.size, self.size, QImage.Format_ARGB32)
            overlay.fill(QColor(0,0,0,0))
            op = QPainter(overlay)
            op.setRenderHint(QPainter.Antialiasing)
            bands = 20
            for i in range(bands):
                alpha = int(12 + 100 * (i % 2 == 0))
                col = QColor(255, 255, 255, alpha) if (i % 3 == 0) else QColor(0, 0, 0, alpha//2)
                op.fillRect(0, int(i*(self.size/bands)), self.size, int(self.size/bands), col)
            op.end()
            p.drawImage(0, 0, overlay)

        p.end()
        return self.img

    def _draw_weft_ribbon(self, p: QPainter, center_y, thickness, cell_w, warp_thickness, row_index, inner, slices):
        half = thickness / 2.0
        path_left = int(self.size * 0.03) - warp_thickness
        path_width = inner + 2*warp_thickness
        slice_w = (path_width) / slices
        amplitude = max(1.0, thickness * 0.12) * (1.0 / max(0.5, self.tension))
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        x0 = path_left
        # draw slices with alternating light/dark for subtle curvature
        for i in range(slices):
            sx = x0 + i * slice_w
            offset = math.sin((i + row_index*3) * freq + row_index) * amplitude
            base_col = QColor(*self.weft_rgb)
            # alternate lighten/darken
            if (i % 8) < 4:
                col = QColor(clamp(base_col.red()+12, 0, 255), clamp(base_col.green()+12, 0, 255), clamp(base_col.blue()+12, 0, 255))
            else:
                col = QColor(clamp(base_col.red()-10, 0, 255), clamp(base_col.green()-10, 0, 255), clamp(base_col.blue()-10, 0, 255))
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(int(sx), int(center_y - half + offset), int(slice_w) + 1, int(thickness) + 1)
        # fiber strokes
        p.setPen(QPen(QColor(0,0,0,30), 1))
        fiber_count = int(max(8, thickness * 0.5))
        for _ in range(fiber_count):
            fx = path_left + random.random() * path_width
            fy = center_y + (random.random() - 0.5) * thickness * 0.6
            lx = fx + (random.random() - 0.5) * thickness * 0.7
            ly = fy + (random.random() - 0.5) * thickness * 0.7
            p.drawLine(int(fx), int(fy), int(lx), int(ly))

    def _draw_warp_ribbon(self, p: QPainter, center_x, thickness, cell_h, weft_thickness, col_index, inner, slices):
        half = thickness / 2.0
        path_top = int(self.size * 0.03) - weft_thickness
        path_height = inner + 2*weft_thickness
        slice_h = (path_height) / slices
        amplitude = max(1.0, thickness * 0.12) * (1.0 / max(0.5, self.tension))
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        y0 = path_top
        for i in range(slices):
            sy = y0 + i * slice_h
            offset = math.sin((i + col_index*2) * freq + col_index) * amplitude
            base_col = QColor(*self.warp_rgb)
            if (i % 8) < 4:
                col = QColor(clamp(base_col.red()+12,0,255), clamp(base_col.green()+12,0,255), clamp(base_col.blue()+12,0,255))
            else:
                col = QColor(clamp(base_col.red()-10,0,255), clamp(base_col.green()-10,0,255), clamp(base_col.blue()-10,0,255))
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(int(center_x - half + offset), int(sy), int(thickness) + 1, int(slice_h) + 1)
        # warp fibers
        p.setPen(QPen(QColor(0,0,0,30), 1))
        fiber_count = int(max(8, thickness * 0.5))
        for _ in range(fiber_count):
            fy = path_top + random.random() * path_height
            fx = center_x + (random.random() - 0.5) * thickness * 0.6
            lx = fx + (random.random() - 0.5) * thickness * 0.6
            ly = fy + (random.random() - 0.5) * thickness * 0.6
            p.drawLine(int(fx), int(fy), int(lx), int(ly))

# -------------------- Main Application --------------------

class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weaver — Enhanced Jacquard Simulator")
        self.resize(1600, 980)

        # Left controls
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Enter binary pattern lines (0/1) or use generators / import.")
        # default small pattern
        self.input_edit.setPlainText("01010101\n10101010\n01010101\n10101010\n01010101\n10101010\n01010101\n10101010")

        # Pattern generator controls
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(4, 256); self.rows_spin.setValue(32)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(4, 256); self.cols_spin.setValue(32)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(['Checkerboard','Stripes Vert','Stripes Horz','Diagonal','Triangles','Concentric','Random'])
        self.randomness_slider = QSlider(Qt.Horizontal); self.randomness_slider.setRange(0,100); self.randomness_slider.setValue(50)
        self.seed_edit = QLineEdit(); self.seed_edit.setPlaceholderText("seed (optional)")
        self.symmetry_chk = QCheckBox("Apply mirror symmetry (vertical)")

        # glitch controls
        self.glitch_slider = QSlider(Qt.Horizontal); self.glitch_slider.setRange(0,100); self.glitch_slider.setValue(0)
        self.smear_slider = QSlider(Qt.Horizontal); self.smear_slider.setRange(0,100); self.smear_slider.setValue(0)
        self.temporal_chk = QCheckBox("Temporal glitch / animate")
        self.anim_speed_slider = QSlider(Qt.Horizontal); self.anim_speed_slider.setRange(10,1000); self.anim_speed_slider.setValue(120)
        self.progressive_chk = QCheckBox("Progressive weave animation")

        # tension / visual style
        self.tension_slider = QSlider(Qt.Horizontal); self.tension_slider.setRange(5,200); self.tension_slider.setValue(100)
        # visual style radio
        style_group = QGroupBox("Visual Style")
        self.style_realistic = QRadioButton("Realistic"); self.style_flat = QRadioButton("Flat"); self.style_hc = QRadioButton("High-contrast")
        self.style_realistic.setChecked(True)
        sg_layout = QVBoxLayout(); sg_layout.addWidget(self.style_realistic); sg_layout.addWidget(self.style_flat); sg_layout.addWidget(self.style_hc)
        style_group.setLayout(sg_layout)

        # color controls
        self.warp_color_btn = QPushButton("Warp color (black)")
        self.weft_color_btn = QPushButton("Weft color (white)")
        self.rand_warp_btn = QPushButton("Randomize Warp")
        self.rand_weft_btn = QPushButton("Randomize Weft")

        # generator / import / save
        self.gen_btn = QPushButton("Generate")
        self.import_btn = QPushButton("Import Image -> Binary")
        self.save_pattern_btn = QPushButton("Save Pattern")
        self.load_pattern_btn = QPushButton("Load Pattern")

        # weave / save image / gif export
        self.weave_btn = QPushButton("Weave Pattern")
        self.save_img_btn = QPushButton("Save Image")
        self.export_gif_btn = QPushButton("Export Animation GIF")

        # Preview + history (right)
        self.image_label = QLabel()
        self.image_label.setFixedSize(900, 900)  # bigger preview
        self.image_label.setStyleSheet("background:#eee;border:1px solid #bbb;")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.history_list = QListWidget()
        self.history_list.setIconSize(QSize(160, 160))
        self.clear_history_btn = QPushButton("Clear History")

        # Layout left column
        left = QVBoxLayout()
        left.addWidget(QLabel("Binary Input / Generated Pattern"))
        left.addWidget(self.input_edit)

        # generator row
        g1 = QHBoxLayout(); g1.addWidget(QLabel("Rows")); g1.addWidget(self.rows_spin); g1.addWidget(QLabel("Cols")); g1.addWidget(self.cols_spin)
        left.addLayout(g1)
        g2 = QHBoxLayout(); g2.addWidget(QLabel("Pattern")); g2.addWidget(self.pattern_combo); g2.addWidget(QLabel("Randomness")); g2.addWidget(self.randomness_slider)
        left.addLayout(g2)
        g3 = QHBoxLayout(); g3.addWidget(QLabel("Seed")); g3.addWidget(self.seed_edit); g3.addWidget(self.symmetry_chk); left.addLayout(g3)
        left.addWidget(self.gen_btn)
        left.addWidget(self.import_btn)

        left.addWidget(QLabel("Glitch % (flip bits)")); left.addWidget(self.glitch_slider)
        left.addWidget(QLabel("Smear % (row shifts)")); left.addWidget(self.smear_slider)
        left.addWidget(self.temporal_chk)
        left.addWidget(QLabel("Temporal speed (ms per step)")); left.addWidget(self.anim_speed_slider)
        left.addWidget(self.progressive_chk)

        left.addWidget(QLabel("Thread tension (visual)")); left.addWidget(self.tension_slider)
        left.addWidget(style_group)

        color_row = QHBoxLayout()
        color_row.addWidget(self.warp_color_btn); color_row.addWidget(self.rand_warp_btn)
        color_row.addWidget(self.weft_color_btn); color_row.addWidget(self.rand_weft_btn)
        left.addLayout(color_row)

        left.addWidget(self.weave_btn)
        left.addWidget(self.save_img_btn)
        left.addWidget(self.export_gif_btn)
        left.addWidget(self.save_pattern_btn)
        left.addWidget(self.load_pattern_btn)
        left.addStretch()

        # right layout
        right = QVBoxLayout()
        right.addWidget(self.image_label, alignment=Qt.AlignCenter)
        right.addWidget(QLabel("History (click thumbnail to preview)"))
        right.addWidget(self.history_list)
        right.addWidget(self.clear_history_btn)
        right.addStretch()

        main = QHBoxLayout()
        main.addLayout(left, stretch=3)
        main.addLayout(right, stretch=4)
        self.setLayout(main)

        # state
        self.warp_rgb = (0,0,0)
        self.weft_rgb = (255,255,255)
        self.grid = parse_input(self.input_edit.toPlainText()) or gen_checkerboard(8,8)
        self.last_image = None
        self.history = []

        # animation state
        self.timer = QTimer()
        self.timer.timeout.connect(self._anim_step)
        self.animating = False
        self.anim_frames = []
        self.anim_grid = None
        self.anim_progress = 0

        # connect signals
        self.gen_btn.clicked.connect(self.on_generate)
        self.import_btn.clicked.connect(self.on_import)
        self.weave_btn.clicked.connect(self.on_weave)
        self.save_img_btn.clicked.connect(self.on_save_image)
        self.export_gif_btn.clicked.connect(self.on_export_gif)
        self.warp_color_btn.clicked.connect(self.on_pick_warp)
        self.weft_color_btn.clicked.connect(self.on_pick_weft)
        self.rand_warp_btn.clicked.connect(self.on_rand_warp)
        self.rand_weft_btn.clicked.connect(self.on_rand_weft)
        self.save_pattern_btn.clicked.connect(self.on_save_pattern)
        self.load_pattern_btn.clicked.connect(self.on_load_pattern)
        self.clear_history_btn.clicked.connect(self.on_clear_history)
        self.history_list.itemClicked.connect(self.on_history_click)

        # initial render
        self.on_weave(initial=True)

    # ---------------- UI actions ----------------

    def on_pick_warp(self):
        col = QColorDialog.getColor(color_tuple_to_qcolor(self.warp_rgb), self, "Choose warp color")
        if col.isValid():
            self.warp_rgb = (col.red(), col.green(), col.blue())
            self.warp_color_btn.setText(f"Warp color ({col.name()})")
            self.on_weave()

    def on_pick_weft(self):
        col = QColorDialog.getColor(color_tuple_to_qcolor(self.weft_rgb), self, "Choose weft color")
        if col.isValid():
            self.weft_rgb = (col.red(), col.green(), col.blue())
            self.weft_color_btn.setText(f"Weft color ({col.name()})")
            self.on_weave()

    def on_rand_warp(self):
        self.warp_rgb = tuple(random.randint(0,255) for _ in range(3))
        self.warp_color_btn.setText(f"Warp color ({QColor(*self.warp_rgb).name()})")
        self.on_weave()

    def on_rand_weft(self):
        self.weft_rgb = tuple(random.randint(0,255) for _ in range(3))
        self.weft_color_btn.setText(f"Weft color ({QColor(*self.weft_rgb).name()})")
        self.on_weave()

    def on_generate(self):
        rows = self.rows_spin.value(); cols = self.cols_spin.value()
        patt = self.pattern_combo.currentText()
        rnd = self.randomness_slider.value() / 100.0
        seed_text = self.seed_edit.text().strip()
        seed = None
        if seed_text != '':
            try: seed = int(seed_text)
            except: seed = abs(hash(seed_text)) % (2**31)
        if patt == 'Checkerboard':
            grid = gen_checkerboard(rows, cols, seed)
        elif patt == 'Stripes Vert':
            period = max(1, 1 + int(rnd * 6))
            grid = gen_stripes(rows, cols, vertical=True, period=period, seed=seed)
        elif patt == 'Stripes Horz':
            period = max(1, 1 + int(rnd * 6))
            grid = gen_stripes(rows, cols, vertical=False, period=period, seed=seed)
        elif patt == 'Diagonal':
            grid = gen_diagonal(rows, cols, seed)
        elif patt == 'Triangles':
            grid = gen_triangles(rows, cols, seed)
        elif patt == 'Concentric':
            rings = max(2, int(2 + rnd*10))
            grid = gen_concentric(rows, cols, rings=rings, seed=seed)
        else:
            grid = gen_random(rows, cols, randomness=1.0-rnd, seed=seed)

        # optional symmetry
        if self.symmetry_chk.isChecked():
            grid = mirror_symmetry(grid, axis='vertical')

        # apply static glitch flips / smear if sliders > 0
        flip = self.glitch_slider.value() / 100.0
        smear = self.smear_slider.value() / 100.0
        if flip > 0.0 or smear > 0.0:
            grid = apply_glitch_static(grid, flip_pct=flip, smear_pct=smear, seed=seed)

        self.grid = grid
        self.input_edit.setPlainText(grid_to_text(grid))
        self.on_weave()

    def on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if not path:
            return
        rows = self.rows_spin.value(); cols = self.cols_spin.value()
        try:
            grid = image_to_binary_grid(path, rows, cols, threshold=128, invert=False)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", f"Could not import image:\n{e}")
            return
        self.grid = grid
        self.input_edit.setPlainText(grid_to_text(grid))
        self.on_weave()

    def _prepare_grid_from_input(self):
        txt = self.input_edit.toPlainText()
        grid = parse_input(txt)
        if not grid:
            return None
        # apply static glitch if requested before rendering
        flip = self.glitch_slider.value() / 100.0
        smear = self.smear_slider.value() / 100.0
        seed_text = self.seed_edit.text().strip()
        seed = None
        if seed_text != '':
            try: seed = int(seed_text)
            except: seed = abs(hash(seed_text)) % (2**31)
        grid = apply_glitch_static(grid, flip_pct=flip, smear_pct=smear, seed=seed)
        return grid

    def on_weave(self, initial=False):
        # stop any running animation
        if self.animating:
            self.timer.stop()
            self.animating = False
            self.anim_frames = []
            self.anim_grid = None

        # determine visual style
        if self.style_flat.isChecked():
            style = 'flat'
        elif self.style_hc.isChecked():
            style = 'high-contrast'
        else:
            style = 'realistic'

        # build grid (pre-glitch applied from input) or from generator
        grid = self._prepare_grid_from_input()
        if grid is None:
            QMessageBox.warning(self, "No data", "Enter or generate a pattern first.")
            return

        # animation vs static
        # if temporal glitch/animate is checked, run temporal mode where bits flip over time
        self.anim_frames = []
        self.anim_progress = 0
        if self.temporal_chk.isChecked():
            # prepare animation grid base
            seed_text = self.seed_edit.text().strip()
            seed = None
            if seed_text != '':
                try: seed = int(seed_text)
                except: seed = abs(hash(seed_text)) % (2**31)
            # copy base
            self.anim_grid = [list(r) for r in grid]
            self.animating = True
            interval = max(10, self.anim_speed_slider.value())
            self.timer.start(interval)
            return  # timer will call _anim_step which renders progressive frames

        # progressive weave (line-by-line)
        if self.progressive_chk.isChecked():
            # render progressive frames quickly and set last frame
            rows = len(grid)
            frames = []
            for progress in range(1, rows+1):
                renderer = WeaverCanvas(grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb,
                                        canvas_size=CANVAS_SIZE, tension=self.tension_slider.value()/100.0,
                                        visual_style=style)
                qimg = renderer.render(partial_progress=progress)
                frames.append(qimg)
            self._set_last_image_and_history(frames[-1])
            # also keep frames for optional GIF export
            self.anim_frames = frames
            return

        # static render
        renderer = WeaverCanvas(grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb,
                                canvas_size=CANVAS_SIZE, tension=self.tension_slider.value()/100.0,
                                visual_style=style)
        qimg = renderer.render()
        self._set_last_image_and_history(qimg)

    def _anim_step(self):
        # Called repeatedly when temporal_chk is on to animate flipping bits progressively
        if not self.animating:
            self.timer.stop()
            return
        if self.anim_grid is None:
            self.timer.stop()
            self.animating = False
            return
        rows = len(self.anim_grid); cols = len(self.anim_grid[0]) if rows else 0
        # flip a few bits each step
        flips = max(1, int(rows*cols*0.004))
        for _ in range(flips):
            r = random.randrange(rows)
            c = random.randrange(cols)
            self.anim_grid[r][c] = 1 - self.anim_grid[r][c]
        # progressive draw amount grows too for weave effect
        self.anim_progress = min(rows, self.anim_progress + max(1, rows//20))
        style = 'flat' if self.style_flat.isChecked() else ('high-contrast' if self.style_hc.isChecked() else 'realistic')
        renderer = WeaverCanvas(self.anim_grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb,
                                canvas_size=CANVAS_SIZE, tension=self.tension_slider.value()/100.0,
                                visual_style=style)
        qimg = renderer.render(partial_progress=self.anim_progress)
        # collect frames for GIF
        self.anim_frames.append(qimg)
        self._set_preview(qimg)
        if self.anim_progress >= rows:
            self.timer.stop()
            self.animating = False
            # final add to history
            self._add_history_frame(qimg)

    # ---------------- Preview / History ----------------

    def _set_preview(self, qimg):
        preview = qimg.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(QPixmap.fromImage(preview))
        self.last_image = qimg

    def _add_history_frame(self, qimg):
        # keep up to 14 thumbnails
        if len(self.history) >= 14:
            self.history.pop(0)
            self.history_list.takeItem(0)
        self.history.append(qimg.copy())
        item = QListWidgetItem()
        icon = QPixmap.fromImage(qimg).scaled(160,160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        item.setIcon(QIcon(icon))
        item.setData(Qt.UserRole, len(self.history)-1)
        self.history_list.addItem(item)

    def _set_last_image_and_history(self, qimg):
        self._set_preview(qimg)
        self._add_history_frame(qimg)

    def on_history_click(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        if idx is None: return
        img = self.history[idx]
        self._set_preview(img)

    def on_clear_history(self):
        self.history = []
        self.history_list.clear()

    # ---------------- Save / Export ----------------

    def on_save_image(self):
        if self.last_image is None:
            QMessageBox.information(self, "No image", "Nothing to save. Press 'Weave Pattern' first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save woven image", "weave.png", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg)")
        if not path:
            return
        # use QImage.save for convenience
        ok = self.last_image.save(path)
        if ok:
            QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")
        else:
            QMessageBox.warning(self, "Save failed", "Could not save image.")

    def on_save_pattern(self):
        txt = self.input_edit.toPlainText()
        if not txt.strip():
            QMessageBox.information(self, "No pattern", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save pattern", "pattern.txt", "Text Files (*.txt)")
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(txt)
        QMessageBox.information(self, "Saved", f"Pattern saved to:\n{path}")

    def on_load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load pattern", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            grid, txt = load_pattern_from_file(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Could not load pattern:\n{e}")
            return
        self.input_edit.setPlainText(txt)
        self.grid = grid
        self.on_weave()

    # ---------------- GIF Export ----------------

    def on_export_gif(self):
        # Export current anim_frames if available (temporal or progressive), otherwise produce frames by progressive build
        frames_qimg = list(self.anim_frames)  # may be empty
        if not frames_qimg:
            # produce progressive frames from current grid
            grid = parse_input(self.input_edit.toPlainText())
            if not grid:
                QMessageBox.warning(self, "No data", "Generate or enter a pattern first.")
                return
            rows = len(grid)
            style = 'flat' if self.style_flat.isChecked() else ('high-contrast' if self.style_hc.isChecked() else 'realistic')
            for progress in range(1, rows+1):
                renderer = WeaverCanvas(grid, warp_rgb=self.warp_rgb, weft_rgb=self.weft_rgb,
                                        canvas_size=CANVAS_SIZE, tension=self.tension_slider.value()/100.0,
                                        visual_style=style)
                qimg = renderer.render(partial_progress=progress)
                frames_qimg.append(qimg)

        # convert QImage frames -> PIL images
        pil_frames = []
        for q in frames_qimg:
            pil = qimage_to_pil(q)
            pil_frames.append(pil.convert("RGBA"))

        if not pil_frames:
            QMessageBox.warning(self, "No frames", "No frames to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Animation GIF", "weave_anim.gif", "GIF Files (*.gif)")
        if not path:
            return

        # save via Pillow (optimize)
        # reduce size for gif to something reasonable
        max_gif_dim = 800
        w, h = pil_frames[0].size
        scale = min(1.0, max_gif_dim / max(w, h))
        if scale < 1.0:
            new_size = (int(w*scale), int(h*scale))
            pil_frames = [f.resize(new_size, Image.LANCZOS) for f in pil_frames]

        try:
            pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], loop=0, duration=max(40, self.anim_speed_slider.value()), disposal=2)
            QMessageBox.information(self, "Saved", f"Animated GIF saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", f"Could not export GIF:\n{e}")

    # ---------------- Helpers for signals ----------------

    def on_save_pattern(self):
        txt = self.input_edit.toPlainText()
        if not txt.strip():
            QMessageBox.information(self, "No pattern", "Nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save pattern", "pattern.txt", "Text Files (*.txt)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(txt)
            QMessageBox.information(self, "Saved", f"Pattern saved to:\n{path}")

    def on_load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load pattern", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            grid, txt = load_pattern_from_file(path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", f"Could not load pattern:\n{e}")
            return
        self.input_edit.setPlainText(txt)
        self.grid = grid
        self.on_weave()

# -------------------- Run App --------------------

def main():
    app = QApplication(sys.argv)
    w = WeavingApp()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
