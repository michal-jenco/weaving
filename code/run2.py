# weaver_full.py
# Jacquard Punch-Card Weaver — full version
# Features:
#  - Static textile-realistic rendering (1200x1200)
#  - Animated weaving (row-by-row)
#  - Load / Save punch-card (.txt) files
#  - Multicolor warp/weft (per-thread/per-row gradient/random)
#  - Perlin-like texture overlay (blendable)
#  - Logical overlay (show 0/1 grid)
#
# Dependencies: PyQt5, Pillow (optional)
# Usage: python weaver_full.py

import sys, math, random, time
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QColor, QLinearGradient, QPixmap, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QColorDialog, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QCheckBox, QSlider, QSpinBox, QComboBox
)

CANVAS_SIZE = 1920

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

# Simple value-noise-based Perlin-like generator (pure python)
def generate_value_noise(width, height, scale=64, octaves=4, persistence=0.5, seed=None):
    if seed is None:
        seed = random.randint(0, 10**9)
    rand = random.Random(seed)
    # base grid size
    def smoothstep(t):
        return t * t * (3 - 2 * t)
    def lerp(a, b, t):
        return a + (b - a) * t

    base = [[rand.random() for _ in range((width // scale) + 3)] for __ in range((height // scale) + 3)]
    def sample(x, y):
        gx = x / scale
        gy = y / scale
        ix = int(math.floor(gx))
        iy = int(math.floor(gy))
        fx = gx - ix
        fy = gy - iy
        fx_s = smoothstep(fx)
        fy_s = smoothstep(fy)
        # fetch four
        v00 = base[iy][ix]
        v10 = base[iy][ix+1]
        v01 = base[iy+1][ix]
        v11 = base[iy+1][ix+1]
        a = lerp(v00, v10, fx_s)
        b = lerp(v01, v11, fx_s)
        return lerp(a, b, fy_s)

    # combine octaves
    img = [[0.0]*width for _ in range(height)]
    amplitude = 1.0
    freq_scale = 1.0
    max_amp = 0.0
    for o in range(octaves):
        for y in range(height):
            for x in range(width):
                img[y][x] += sample(x * freq_scale, y * freq_scale) * amplitude
        max_amp += amplitude
        amplitude *= persistence
        freq_scale *= 2.0
    # normalize and convert 0..255
    result = [ [ int(255 * (img[y][x] / max_amp)) for x in range(width) ] for y in range(height) ]
    return result

def color_rgb_to_qcolor(rgb):
    return QColor(rgb[0], rgb[1], rgb[2])

def lerp_color(c1, c2, t):
    return (int(c1[0] + (c2[0]-c1[0])*t),
            int(c1[1] + (c2[1]-c1[1])*t),
            int(c1[2] + (c2[2]-c1[2])*t))

class WeaverCanvas:
    def __init__(self, grid, warp_base=(0,0,0), weft_base=(255,255,255),
                 canvas_size=CANVAS_SIZE, multicolor_warp=False, multicolor_weft=False,
                 warp_palette=None, weft_palette=None,
                 texture_strength=0.25, texture_seed=None,
                 show_logic=False, animate_rows=0):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.warp_base = warp_base
        self.weft_base = weft_base
        self.size = canvas_size
        self.multicolor_warp = multicolor_warp
        self.multicolor_weft = multicolor_weft
        self.warp_palette = warp_palette
        self.weft_palette = weft_palette
        self.texture_strength = texture_strength
        self.texture_seed = texture_seed
        self.show_logic = show_logic
        self.animate_rows = animate_rows  # if >0, show only first animate_rows rows (from top)
        self.img = QImage(self.size, self.size, QImage.Format_RGB32)
        self.img.fill(QColor(240,240,240))
        self.painter = QPainter(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing)
        self.margin = int(self.size * 0.03)
        self.inner = self.size - 2*self.margin

    def _warp_color_for_col(self, col_idx):
        if not self.multicolor_warp or not self.warp_palette:
            return self.warp_base
        # interpolate across palette along columns
        t = col_idx / max(1, self.cols-1)
        # palette is list: blend across segments
        n = len(self.warp_palette)
        if n == 1:
            return self.warp_palette[0]
        seg = t * (n-1)
        i = int(math.floor(seg))
        frac = seg - i
        return lerp_color(self.warp_palette[i], self.warp_palette[min(i+1, n-1)], frac)

    def _weft_color_for_row(self, row_idx):
        if not self.multicolor_weft or not self.weft_palette:
            return self.weft_base
        t = row_idx / max(1, self.rows-1)
        n = len(self.weft_palette)
        if n == 1:
            return self.weft_palette[0]
        seg = t * (n-1)
        i = int(math.floor(seg))
        frac = seg - i
        return lerp_color(self.weft_palette[i], self.weft_palette[min(i+1, n-1)], frac)

    def render(self):
        if self.rows == 0 or self.cols == 0:
            self.painter.fillRect(0,0,self.size,self.size, QColor(240,240,240))
            self.painter.end()
            return self.img

        cell_h = self.inner / self.rows
        cell_w = self.inner / self.cols
        warp_thickness = max(6, int(cell_w * 0.9))
        weft_thickness = max(6, int(cell_h * 0.9))

        area = QRectF(self.margin, self.margin, self.inner, self.inner)
        self.painter.fillRect(area, QColor(225,225,225))

        # draw base weft ribbons (only rows up to animate_rows if animation used)
        rows_to_draw = self.rows if self.animate_rows <= 0 else min(self.animate_rows, self.rows)
        for r in range(rows_to_draw):
            y = self.margin + r*cell_h + cell_h/2.0
            c = self._weft_color_for_row(r)
            self._draw_weft_ribbon(y, weft_thickness, cell_w, warp_thickness, r, c)

        # draw warp ribbons for all columns (they may cover wefts). If animation is enabled
        # but we only drew some wefts above, we still draw warps fully so they appear as weaving progresses.
        for c in range(self.cols):
            x = self.margin + c*cell_w + cell_w/2.0
            col_color = self._warp_color_for_col(c)
            self._draw_warp_ribbon(x, warp_thickness, cell_h, weft_thickness, c, col_color)

        # texture overlay
        if self.texture_strength and self.texture_strength > 0.01:
            self._apply_texture()

        # logic overlay
        if self.show_logic:
            self._draw_logic_overlay(cell_w, cell_h)

        self.painter.end()
        return self.img

    def _draw_weft_ribbon(self, center_y, thickness, cell_w, warp_thickness, row_index, color_rgb):
        half = thickness/2.0
        path_left = self.margin - warp_thickness
        path_right = self.margin + self.inner + warp_thickness
        path_rect = QRectF(path_left, center_y-half, path_right-path_left, thickness)
        grad = QLinearGradient(path_rect.left(), path_rect.top(), path_rect.right(), path_rect.top())
        base = QColor(*color_rgb)
        dark = QColor(max(0,color_rgb[0]-36), max(0,color_rgb[1]-36), max(0,color_rgb[2]-36))
        light = QColor(min(255,color_rgb[0]+36), min(255,color_rgb[1]+36), min(255,color_rgb[2]+36))
        grad.setColorAt(0.0, dark)
        grad.setColorAt(0.5, light)
        grad.setColorAt(1.0, dark)
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(QBrush(grad))
        slices = 140
        slice_w = (path_rect.width()) / slices
        amplitude = max(1.0, thickness * 0.12)
        freq = 2.0 * math.pi / max(120.0, slices/2.0)
        x0 = path_rect.left()
        for i in range(slices):
            sx = x0 + i*slice_w
            offset = math.sin((i + row_index*3) * freq + row_index) * amplitude
            sr = QRectF(sx, path_rect.top()+offset, slice_w+1, path_rect.height())
            self.painter.drawRect(sr)
        # fiber strokes
        self.painter.setPen(QPen(QColor(0,0,0,22), 1))
        fiber_count = int(max(18, thickness*0.6))
        for f in range(fiber_count):
            fx = path_rect.left() + random.random()*path_rect.width()
            fy = center_y + (random.random()-0.5)*thickness*0.6
            lx = fx + (random.random()-0.5)*thickness*0.6
            ly = fy + (random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy), QPointF(lx,ly))

    def _draw_warp_ribbon(self, center_x, thickness, cell_h, weft_thickness, col_index, color_rgb):
        half = thickness/2.0
        path_top = self.margin - weft_thickness
        path_bottom = self.margin + self.inner + weft_thickness
        path_rect = QRectF(center_x-half, path_top, thickness, path_bottom-path_top)
        grad = QLinearGradient(path_rect.left(), path_rect.top(), path_rect.left(), path_rect.bottom())
        dark = QColor(max(0,color_rgb[0]-36), max(0,color_rgb[1]-36), max(0,color_rgb[2]-36))
        light = QColor(min(255,color_rgb[0]+36), min(255,color_rgb[1]+36), min(255,color_rgb[2]+36))
        grad.setColorAt(0.0, dark)
        grad.setColorAt(0.5, light)
        grad.setColorAt(1.0, dark)
        self.painter.setBrush(QBrush(grad))
        self.painter.setPen(Qt.NoPen)
        slices = 140
        slice_h = (path_rect.height()) / slices
        amplitude = max(1.0, thickness*0.12)
        freq = 2.0*math.pi / max(120.0, slices/2.0)
        y0 = path_rect.top()
        for i in range(slices):
            sy = y0 + i*slice_h
            offset = math.sin((i + col_index*2) * freq + col_index) * amplitude
            sr = QRectF(path_rect.left()+offset, sy, path_rect.width(), slice_h+1)
            self.painter.drawRect(sr)
        # fiber strokes
        self.painter.setPen(QPen(QColor(0,0,0,28), 1))
        fiber_count = int(max(18, thickness*0.55))
        for f in range(fiber_count):
            fy = path_rect.top() + random.random()*path_rect.height()
            fx = center_x + (random.random()-0.5)*thickness*0.6
            lx = fx + (random.random()-0.5)*thickness*0.6
            ly = fy + (random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy), QPointF(lx,ly))

        # interlacing: if grid[row][col] == 0 (warp under), draw small weft cap on top
        cell_h = self.inner / self.rows
        cell_w = self.inner / self.cols
        half_weft = weft_thickness/2.0
        for r in range(self.rows):
            # if animate_rows active and this row wasn't drawn as weft beneath, that means the warp should appear above until weft is added.
            if self.animate_rows > 0 and r >= self.animate_rows:
                # treat as warp over (no cap, because weft isn't present yet)
                continue
            bit = self.grid[r][col_index]
            y = self.margin + r*cell_h + cell_h/2.0
            x = center_x
            if bit == 0:
                # draw weft cap
                rect = QRectF(x - cell_w/2.0 - thickness/2.0, y - half_weft, cell_w + thickness, weft_thickness)
                grad_local = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
                grad_local.setColorAt(0.0, QColor(max(0,self.weft_base[0]-30), max(0,self.weft_base[1]-30), max(0,self.weft_base[2]-30)))
                grad_local.setColorAt(0.5, QColor(min(255,self.weft_base[0]+30), min(255,self.weft_base[1]+30), min(255,self.weft_base[2]+30)))
                grad_local.setColorAt(1.0, QColor(max(0,self.weft_base[0]-30), max(0,self.weft_base[1]-30), max(0,self.weft_base[2]-30)))
                self.painter.setBrush(QBrush(grad_local))
                self.painter.setPen(Qt.NoPen)
                self.painter.drawRoundedRect(rect, 3, 3)

    def _apply_texture(self):
        # generate noise sized same as canvas interior and blend with multiply/overlay
        w = int(self.inner)
        h = int(self.inner)
        # use the same seed for consistency if provided
        seed = self.texture_seed if self.texture_seed is not None else int(time.time() % 1e9)
        noise = generate_value_noise(w, h, scale=max(24, int(w/24)), octaves=4, persistence=0.5, seed=seed)
        # blend: for each pixel inside margin, sample noise and darken/lighten by texture_strength
        img_bits = self.img.bits()
        img_bits.setsize(self.img.byteCount())
        # iterate pixels (slower but fine for 1200x1200)
        for yy in range(h):
            for xx in range(w):
                global_x = int(self.margin + xx)
                global_y = int(self.margin + yy)
                base_col = QColor(self.img.pixel(global_x, global_y))
                n = noise[yy][xx] / 255.0  # 0..1
                # subtle overlay using lerp to darker or lighter
                strength = self.texture_strength
                # propose to slightly darken midtones where n < 0.5, lighten where >0.5
                if n < 0.5:
                    t = (0.5 - n) * 2.0 * strength
                    new_r = int(base_col.red() * (1 - t))
                    new_g = int(base_col.green() * (1 - t))
                    new_b = int(base_col.blue() * (1 - t))
                else:
                    t = (n - 0.5) * 2.0 * strength
                    new_r = int(base_col.red() + (255 - base_col.red()) * t * 0.25)
                    new_g = int(base_col.green() + (255 - base_col.green()) * t * 0.25)
                    new_b = int(base_col.blue() + (255 - base_col.blue()) * t * 0.25)
                self.img.setPixel(global_x, global_y, QColor(new_r, new_g, new_b).rgb())

    def _draw_logic_overlay(self, cell_w, cell_h):
        # draw semi-transparent grid and 0/1 digits at crossing centers
        pen = QPen(QColor(0,0,0,80), 1)
        self.painter.setPen(pen)
        # vertical lines
        for c in range(self.cols+1):
            x = self.margin + c*cell_w
            self.painter.drawLine(QPointF(x, self.margin), QPointF(x, self.margin+self.inner))
        # horizontal lines
        for r in range(self.rows+1):
            y = self.margin + r*cell_h
            self.painter.drawLine(QPointF(self.margin, y), QPointF(self.margin+self.inner, y))
        # draw digits
        font = self.painter.font()
        font.setPointSize(max(8, int(min(cell_w, cell_h)*0.35)))
        self.painter.setFont(font)
        for r in range(self.rows):
            for c in range(self.cols):
                bit = self.grid[r][c]
                x = self.margin + c*cell_w + cell_w/2.0
                y = self.margin + r*cell_h + cell_h/2.0
                txt = "1" if bit==1 else "0"
                # draw with subtle halo
                self.painter.setPen(QPen(QColor(255,255,255,220), 3))
                self.painter.drawText(QRectF(x-20, y-12, 40, 24), Qt.AlignCenter, txt)
                self.painter.setPen(QPen(QColor(0,0,0,200), 1))
                self.painter.drawText(QRectF(x-20, y-12, 40, 24), Qt.AlignCenter, txt)

class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jacquard Weaver — Full")
        self.resize(1500, 920)
        # Widgets
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Binary punch-card lines. One row per line.")
        self.input_edit.setPlainText("0101010101\n1010101010\n0101010101\n1010101010\n0101010101\n1010101010\n0101010101\n1010101010\n0101010101\n1010101010")
        self.warp_color_btn = QPushButton("Warp color")
        self.weft_color_btn = QPushButton("Weft color")
        self.load_btn = QPushButton("Load .txt")
        self.save_pattern_btn = QPushButton("Save .txt")
        self.weave_btn = QPushButton("Weave (Render)")
        self.save_img_btn = QPushButton("Save Image")
        # switches
        self.animate_cb = QCheckBox("Animated weaving")
        self.animate_speed_label = QLabel("Speed")
        self.animate_speed = QSpinBox(); self.animate_speed.setRange(10,1000); self.animate_speed.setValue(150)
        self.multicolor_warp_cb = QCheckBox("Multicolor warp")
        self.multicolor_weft_cb = QCheckBox("Multicolor weft")
        self.texture_cb = QCheckBox("Texture overlay")
        self.texture_strength_slider = QSlider(Qt.Horizontal); self.texture_strength_slider.setRange(0,100); self.texture_strength_slider.setValue(20)
        self.show_logic_cb = QCheckBox("Show logic (0/1 overlay)")
        self.randomize_colors_btn = QPushButton("Randomize palettes")
        self.preview_label = QLabel(); self.preview_label.setFixedSize(CANVAS_SIZE//2, CANVAS_SIZE//2)
        self.preview_label.setStyleSheet("background: #eee; border: 1px solid #ccc;")
        # palette style selection
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["Gradient", "Rainbow", "Monochrome", "Warm", "Cool"])

        # colors default
        self.warp_rgb = (0,0,0)
        self.weft_rgb = (255,255,255)
        self.warp_palette = [self.warp_rgb]
        self.weft_palette = [self.weft_rgb]

        # layout
        left = QVBoxLayout()
        left.addWidget(self.input_edit)
        rowA = QHBoxLayout()
        rowA.addWidget(self.warp_color_btn)
        rowA.addWidget(self.weft_color_btn)
        left.addLayout(rowA)
        rowB = QHBoxLayout()
        rowB.addWidget(self.load_btn)
        rowB.addWidget(self.save_pattern_btn)
        left.addLayout(rowB)
        left.addWidget(self.weave_btn)
        left.addWidget(self.save_img_btn)
        left.addWidget(self.randomize_colors_btn)
        left.addWidget(self.palette_combo)
        left.addStretch()

        right = QVBoxLayout()
        right.addWidget(self.preview_label, alignment=Qt.AlignCenter)
        switches = QVBoxLayout()
        switches.addWidget(self.animate_cb)
        sp = QHBoxLayout(); sp.addWidget(self.animate_speed_label); sp.addWidget(self.animate_speed)
        switches.addLayout(sp)
        switches.addWidget(self.multicolor_warp_cb)
        switches.addWidget(self.multicolor_weft_cb)
        switches.addWidget(self.texture_cb)
        trow = QHBoxLayout(); trow.addWidget(QLabel("Strength")); trow.addWidget(self.texture_strength_slider)
        switches.addLayout(trow)
        switches.addWidget(self.show_logic_cb)
        right.addLayout(switches)
        right.addStretch()

        main = QHBoxLayout()
        main.addLayout(left, stretch=2)
        main.addLayout(right, stretch=1)
        self.setLayout(main)

        # connections
        self.warp_color_btn.clicked.connect(self.pick_warp_color)
        self.weft_color_btn.clicked.connect(self.pick_weft_color)
        self.load_btn.clicked.connect(self.load_pattern)
        self.save_pattern_btn.clicked.connect(self.save_pattern)
        self.weave_btn.clicked.connect(self.on_weave)
        self.save_img_btn.clicked.connect(self.save_image)
        self.randomize_colors_btn.clicked.connect(self.randomize_palettes)
        self.animate_cb.toggled.connect(self.on_animate_toggle)
        self.multicolor_warp_cb.toggled.connect(lambda v: None)
        self.multicolor_weft_cb.toggled.connect(lambda v: None)
        self.texture_cb.toggled.connect(lambda v: None)
        self.show_logic_cb.toggled.connect(lambda v: None)
        self.palette_combo.currentIndexChanged.connect(lambda i: None)

        # animation timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._animation_step)
        self.animation_state = {"current_rows": 0, "max_rows": 0}

        # initial render
        self.last_image = None
        self.on_weave()

    def pick_warp_color(self):
        col = QColorDialog.getColor(QColor(*self.warp_rgb), self, "Choose warp color")
        if col.isValid():
            self.warp_rgb = (col.red(), col.green(), col.blue())
            self.warp_palette = [self.warp_rgb]
            self.on_weave()

    def pick_weft_color(self):
        col = QColorDialog.getColor(QColor(*self.weft_rgb), self, "Choose weft color")
        if col.isValid():
            self.weft_rgb = (col.red(), col.green(), col.blue())
            self.weft_palette = [self.weft_rgb]
            self.on_weave()

    def load_pattern(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open punch card file", "", "Text Files (*.txt);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
            self.input_edit.setPlainText(txt)
            self.on_weave()
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))

    def save_pattern(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save punch card file", "pattern.txt", "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.input_edit.toPlainText())
            QMessageBox.information(self, "Saved", "Pattern saved.")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def randomize_palettes(self):
        style = self.palette_combo.currentText()
        # generate 3-color palette for warp and weft depending on style
        def make_palette(style):
            if style == "Rainbow":
                return [(255,0,0),(0,200,80),(0,90,255)]
            if style == "Warm":
                return [(120,10,10),(220,80,20),(250,180,45)]
            if style == "Cool":
                return [(10,30,120),(10,140,180),(90,200,190)]
            if style == "Monochrome":
                g = random.randint(30,220)
                return [(g,g,g)]
            # Gradient or default
            return [ (random.randint(20,230), random.randint(20,230), random.randint(20,230)),
                     (random.randint(20,230), random.randint(20,230), random.randint(20,230)) ]
        self.warp_palette = make_palette(style)
        self.weft_palette = make_palette(style if style!="Rainbow" else "Cool")
        # ensure warp/weft bases exist for non-multicolor fallback
        if not self.multicolor_warp_cb.isChecked():
            self.warp_rgb = self.warp_palette[0]
        if not self.multicolor_weft_cb.isChecked():
            self.weft_rgb = self.weft_palette[0]
        self.on_weave()

    def on_animate_toggle(self, checked):
        if checked:
            # start animation
            grid = parse_input(self.input_edit.toPlainText())
            if not grid:
                QMessageBox.warning(self, "Empty", "Enter a pattern first.")
                self.animate_cb.setChecked(False)
                return
            self.animation_state["current_rows"] = 0
            self.animation_state["max_rows"] = len(grid)
            interval = max(10, self.animate_speed.value())
            self.timer.start(interval)
        else:
            self.timer.stop()

    def _animation_step(self):
        self.animation_state["current_rows"] += 1
        if self.animation_state["current_rows"] > self.animation_state["max_rows"]:
            self.timer.stop()
            self.animate_cb.setChecked(False)
            return
        self.on_weave(animate_rows=self.animation_state["current_rows"], skip_timer_control=True)

    def on_weave(self, animate_rows=0, skip_timer_control=False):
        grid = parse_input(self.input_edit.toPlainText())
        if not grid:
            QMessageBox.warning(self, "No pattern", "Enter binary lines first.")
            return
        # if animation active and not provided animate_rows param, we let timer handle steps
        if self.animate_cb.isChecked() and animate_rows == 0 and not skip_timer_control:
            # when toggled to on, timer will call steps
            return
        animate_rows_param = animate_rows if animate_rows>0 else (len(grid) if not self.animate_cb.isChecked() else 0)
        # collect settings
        multicolor_warp = self.multicolor_warp_cb.isChecked()
        multicolor_weft = self.multicolor_weft_cb.isChecked()
        texture_on = self.texture_cb.isChecked()
        texture_strength = self.texture_strength_slider.value() / 100.0
        show_logic = self.show_logic_cb.isChecked()
        # if no palettes assigned, set from bases
        if not self.warp_palette:
            self.warp_palette = [self.warp_rgb]
        if not self.weft_palette:
            self.weft_palette = [self.weft_rgb]
        renderer = WeaverCanvas(grid,
                                warp_base=self.warp_rgb, weft_base=self.weft_rgb,
                                canvas_size=CANVAS_SIZE,
                                multicolor_warp=multicolor_warp, multicolor_weft=multicolor_weft,
                                warp_palette=self.warp_palette, weft_palette=self.weft_palette,
                                texture_strength=texture_strength if texture_on else 0.0,
                                texture_seed=None,
                                show_logic=show_logic,
                                animate_rows=animate_rows_param)
        qimg = renderer.render()
        self.last_image = qimg
        preview = qimg.scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(QPixmap.fromImage(preview))

    def save_image(self):
        if self.last_image is None:
            QMessageBox.information(self, "No image", "Render first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save woven image", "weave.png", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if not path:
            return
        ok = self.last_image.save(path)
        if ok:
            QMessageBox.information(self, "Saved", f"Saved to {path}")
        else:
            QMessageBox.warning(self, "Save failed", "Could not save image.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = WeavingApp()
    w.show()
    sys.exit(app.exec_())
