# weaver_full_expanded.py
# Full Jacquard Punch-Card Weaver with styles, multicolor, texture, animation
# Dependencies: PyQt5, Pillow optional
# Usage: python weaver_full_expanded.py

import sys, math, random, time
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QColor, QLinearGradient, QPixmap, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QColorDialog, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QCheckBox, QSlider, QSpinBox, QComboBox
)

CANVAS_SIZE = 1920

# ---- Utilities ----
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

def generate_value_noise(width, height, scale=64, octaves=4, persistence=0.5, seed=None):
    if seed is None:
        seed = random.randint(0, 10**9)
    rand = random.Random(seed)
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
        v00 = base[iy][ix]
        v10 = base[iy][ix+1]
        v01 = base[iy+1][ix]
        v11 = base[iy+1][ix+1]
        a = lerp(v00, v10, fx_s)
        b = lerp(v01, v11, fx_s)
        return lerp(a, b, fy_s)
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
    result = [ [ int(255 * (img[y][x] / max_amp)) for x in range(width) ] for y in range(height) ]
    return result

def lerp_color(c1, c2, t):
    return (int(c1[0] + (c2[0]-c1[0])*t),
            int(c1[1] + (c2[1]-c1[1])*t),
            int(c1[2] + (c2[2]-c1[2])*t))

# ---- Weaver Canvas ----
class WeaverCanvas:
    def __init__(self, grid, warp_base=(0,0,0), weft_base=(255,255,255),
                 canvas_size=CANVAS_SIZE, multicolor_warp=False, multicolor_weft=False,
                 warp_palette=None, weft_palette=None,
                 texture_strength=0.25, texture_seed=None,
                 show_logic=False, animate_rows=0, style="Gradient"):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.warp_base = warp_base
        self.weft_base = weft_base
        self.size = canvas_size
        self.multicolor_warp = multicolor_warp
        self.multicolor_weft = multicolor_weft
        self.warp_palette = warp_palette or [warp_base]
        self.weft_palette = weft_palette or [weft_base]
        self.texture_strength = texture_strength
        self.texture_seed = texture_seed
        self.show_logic = show_logic
        self.animate_rows = animate_rows
        self.style = style
        self.img = QImage(self.size, self.size, QImage.Format_RGB32)
        self.img.fill(QColor(240,240,240))
        self.painter = QPainter(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing)
        self.margin = int(self.size * 0.03)
        self.inner = self.size - 2*self.margin

    def _warp_color_for_col(self, col_idx):
        if not self.multicolor_warp or not self.warp_palette:
            return self.warp_base
        t = col_idx / max(1, self.cols-1)
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

    def _apply_style_modifiers(self, color_rgb):
        # Apply style modifications
        r,g,b = color_rgb
        if self.style == "Rainbow":
            return ((r+50)%256, (g+80)%256, (b+120)%256)
        elif self.style == "Warm":
            return (min(255, r+40), min(255, g+10), min(255, b+5))
        elif self.style == "Cool":
            return (max(0,r-20), max(0,g+20), min(255,b+60))
        elif self.style == "Monochrome":
            gray = int((r+g+b)/3)
            return (gray, gray, gray)
        return color_rgb

    def render(self):
        if self.rows==0 or self.cols==0:
            self.painter.fillRect(0,0,self.size,self.size, QColor(240,240,240))
            self.painter.end()
            return self.img
        cell_h = self.inner / self.rows
        cell_w = self.inner / self.cols
        warp_thickness = max(6, int(cell_w * 0.9))
        weft_thickness = max(6, int(cell_h * 0.9))
        area = QRectF(self.margin, self.margin, self.inner, self.inner)
        self.painter.fillRect(area, QColor(225,225,225))
        rows_to_draw = self.rows if self.animate_rows <= 0 else min(self.animate_rows, self.rows)
        for r in range(rows_to_draw):
            y = self.margin + r*cell_h + cell_h/2.0
            c = self._weft_color_for_row(r)
            c = self._apply_style_modifiers(c)
            self._draw_weft_ribbon(y, weft_thickness, cell_w, warp_thickness, r, c)
        for c in range(self.cols):
            x = self.margin + c*cell_w + cell_w/2.0
            col_color = self._warp_color_for_col(c)
            col_color = self._apply_style_modifiers(col_color)
            self._draw_warp_ribbon(x, warp_thickness, cell_h, weft_thickness, c, col_color)
        if self.texture_strength>0.01:
            self._apply_texture()
        if self.show_logic:
            self._draw_logic_overlay(cell_w, cell_h)
        self.painter.end()
        return self.img

    # --- Drawing methods ---
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
        slice_w = path_rect.width() / slices
        amplitude = max(1.0, thickness*0.12)
        freq = 2*math.pi/max(120.0, slices/2.0)
        x0 = path_rect.left()
        for i in range(slices):
            sx = x0 + i*slice_w
            offset = math.sin((i + row_index*3) * freq + row_index) * amplitude
            sr = QRectF(sx, path_rect.top()+offset, slice_w+1, path_rect.height())
            self.painter.drawRect(sr)
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
        slice_h = path_rect.height() / slices
        amplitude = max(1.0, thickness*0.12)
        freq = 2*math.pi/max(120.0, slices/2.0)
        y0 = path_rect.top()
        for i in range(slices):
            sy = y0 + i*slice_h
            offset = math.sin((i + col_index*2) * freq + col_index) * amplitude
            sr = QRectF(path_rect.left()+offset, sy, path_rect.width(), slice_h+1)
            self.painter.drawRect(sr)
        self.painter.setPen(QPen(QColor(0,0,0,28), 1))
        fiber_count = int(max(18, thickness*0.55))
        for f in range(fiber_count):
            fy = path_rect.top() + random.random()*path_rect.height()
            fx = center_x + (random.random()-0.5)*thickness*0.6
            lx = fx + (random.random()-0.5)*thickness*0.6
            ly = fy + (random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy), QPointF(lx,ly))
        cell_h = self.inner/self.rows
        cell_w = self.inner/self.cols
        half_weft = weft_thickness/2.0
        for r in range(self.rows):
            if self.animate_rows>0 and r>=self.animate_rows:
                continue
            bit = self.grid[r][col_index]
            y = self.margin + r*cell_h + cell_h/2.0
            x = center_x
            if bit==0:
                rect = QRectF(x - cell_w/2.0 - thickness/2.0, y - half_weft, cell_w + thickness, weft_thickness)
                grad_local = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
                grad_local.setColorAt(0.0, QColor(max(0,self.weft_base[0]-30), max(0,self.weft_base[1]-30), max(0,self.weft_base[2]-30)))
                grad_local.setColorAt(0.5, QColor(min(255,self.weft_base[0]+30), min(255,self.weft_base[1]+30), min(255,self.weft_base[2]+30)))
                grad_local.setColorAt(1.0, QColor(max(0,self.weft_base[0]-30), max(0,self.weft_base[1]-30), max(0,self.weft_base[2]-30)))
                self.painter.setBrush(QBrush(grad_local))
                self.painter.setPen(Qt.NoPen)
                self.painter.drawRoundedRect(rect,3,3)

    def _apply_texture(self):
        w,h = int(self.inner), int(self.inner)
        seed = self.texture_seed if self.texture_seed else int(time.time()%1e9)
        noise = generate_value_noise(w,h,scale=max(24,int(w/24)), octaves=4, persistence=0.5, seed=seed)
        img_bits = self.img.bits(); img_bits.setsize(self.img.byteCount())
        for yy in range(h):
            for xx in range(w):
                global_x = int(self.margin+xx)
                global_y = int(self.margin+yy)
                base_col = QColor(self.img.pixel(global_x, global_y))
                n = noise[yy][xx]/255.0
                s = self.texture_strength
                if n<0.5:
                    t = (0.5-n)*2.0*s
                    new_r = int(base_col.red()*(1-t))
                    new_g = int(base_col.green()*(1-t))
                    new_b = int(base_col.blue()*(1-t))
                else:
                    t = (n-0.5)*2.0*s
                    new_r = int(base_col.red() + (255-base_col.red())*t*0.25)
                    new_g = int(base_col.green() + (255-base_col.green())*t*0.25)
                    new_b = int(base_col.blue() + (255-base_col.blue())*t*0.25)
                self.img.setPixel(global_x, global_y, QColor(new_r,new_g,new_b).rgb())

    def _draw_logic_overlay(self, cell_w, cell_h):
        pen = QPen(QColor(0,0,0,80),1)
        self.painter.setPen(pen)
        for c in range(self.cols+1):
            x = self.margin + c*cell_w
            self.painter.drawLine(QPointF(x,self.margin), QPointF(x,self.margin+self.inner))
        for r in range(self.rows+1):
            y = self.margin + r*cell_h
            self.painter.drawLine(QPointF(self.margin,y), QPointF(self.margin+self.inner, y))
        font = self.painter.font()
        font.setPointSize(max(8,int(min(cell_w,cell_h)*0.35)))
        self.painter.setFont(font)
        for r in range(self.rows):
            for c in range(self.cols):
                bit = self.grid[r][c]
                x = self.margin + c*cell_w + cell_w/2.0
                y = self.margin + r*cell_h + cell_h/2.0
                txt = "1" if bit==1 else "0"
                self.painter.setPen(QPen(QColor(255,255,255,220),3))
                self.painter.drawText(QRectF(x-20,y-12,40,24), Qt.AlignCenter, txt)
                self.painter.setPen(QPen(QColor(0,0,0,200),1))
                self.painter.drawText(QRectF(x-20,y-12,40,24), Qt.AlignCenter, txt)

# ---- Main Application ----
class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jacquard Weaver — Full Expanded")
        self.resize(1500,920)
        # Widgets
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Binary punch-card lines. One row per line.")
        self.input_edit.setPlainText("11111111\n11100111\n11111111\n11111111\n11111111\n11111111\n11111111\n11111111")
        self.render_btn = QPushButton("Render")
        self.save_btn = QPushButton("Save Image")
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Gradient","Rainbow","Monochrome","Warm","Cool"])
        self.multicolor_warp_cb = QCheckBox("Multicolor Warp")
        self.multicolor_weft_cb = QCheckBox("Multicolor Weft")
        self.show_logic_cb = QCheckBox("Show Logic Overlay")
        self.texture_slider = QSlider(Qt.Horizontal)
        self.texture_slider.setMinimum(0); self.texture_slider.setMaximum(100); self.texture_slider.setValue(25)
        self.animate_spin = QSpinBox(); self.animate_spin.setMinimum(0); self.animate_spin.setMaximum(64); self.animate_spin.setValue(0)
        # Layouts
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Punch Card Input"))
        left_layout.addWidget(self.input_edit)
        left_layout.addWidget(self.render_btn)
        left_layout.addWidget(self.save_btn)
        left_layout.addWidget(QLabel("Render Style"))
        left_layout.addWidget(self.style_combo)
        left_layout.addWidget(self.multicolor_warp_cb)
        left_layout.addWidget(self.multicolor_weft_cb)
        left_layout.addWidget(self.show_logic_cb)
        left_layout.addWidget(QLabel("Texture Strength"))
        left_layout.addWidget(self.texture_slider)
        left_layout.addWidget(QLabel("Animate Rows (0=All)"))
        left_layout.addWidget(self.animate_spin)
        self.image_label = QLabel()
        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout,1)
        main_layout.addWidget(self.image_label,3)
        self.setLayout(main_layout)
        # Signals
        self.render_btn.clicked.connect(self.render_canvas)
        self.save_btn.clicked.connect(self.save_image)
        # Colors
        self.warp_palette = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(6)]
        self.weft_palette = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in range(6)]
        self.canvas_img = None

    def render_canvas(self):
        grid = parse_input(self.input_edit.toPlainText())
        style = self.style_combo.currentText()
        canvas = WeaverCanvas(
            grid,
            warp_base=(50,50,50),
            weft_base=(220,220,220),
            multicolor_warp=self.multicolor_warp_cb.isChecked(),
            multicolor_weft=self.multicolor_weft_cb.isChecked(),
            warp_palette=self.warp_palette,
            weft_palette=self.weft_palette,
            texture_strength=self.texture_slider.value()/100.0,
            show_logic=self.show_logic_cb.isChecked(),
            animate_rows=self.animate_spin.value(),
            style=style
        )
        self.canvas_img = canvas.render()
        pixmap = QPixmap.fromImage(self.canvas_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_image(self):
        if self.canvas_img is None:
            QMessageBox.warning(self,"Warning","No rendered image to save!")
            return
        filename,_ = QFileDialog.getSaveFileName(self,"Save Image","","PNG Files (*.png);;JPEG Files (*.jpg)")
        if filename:
            self.canvas_img.save(filename)

# ---- Main Loop ----
if __name__=="__main__":
    app = QApplication(sys.argv)
    win = WeavingApp()
    win.show()
    sys.exit(app.exec_())
