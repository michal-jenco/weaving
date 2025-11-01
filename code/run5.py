# weaver_full_styles.py
# Jacquard Punch-Card Weaver — full version with palette styles
# Features:
#  - Static textile-realistic rendering (1200x1200)
#  - Animated weaving (row-by-row)
#  - Load / Save punch-card (.txt) files
#  - Multicolor warp/weft (per-thread/per-row gradient/random)
#  - Perlin-like texture overlay (blendable)
#  - Logical overlay (show 0/1 grid)
#  - Palette styles: Gradient, Rainbow, Monochrome, Warm, Cool
#
# Dependencies: PyQt5
# Usage: python weaver_full_styles.py

import sys, math, random, time
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QColor, QLinearGradient, QPixmap, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QColorDialog, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QCheckBox, QSlider, QSpinBox, QComboBox
)

CANVAS_SIZE = 1200

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

# Simple value-noise-based Perlin-like generator
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
    result = [[int(255*(img[y][x]/max_amp)) for x in range(width)] for y in range(height)]
    return result

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
        self.animate_rows = animate_rows
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
        seg = t*(n-1)
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
        seg = t*(n-1)
        i = int(math.floor(seg))
        frac = seg - i
        return lerp_color(self.weft_palette[i], self.weft_palette[min(i+1, n-1)], frac)

    def render(self):
        if self.rows==0 or self.cols==0:
            self.painter.fillRect(0,0,self.size,self.size, QColor(240,240,240))
            self.painter.end()
            return self.img
        cell_h = self.inner/self.rows
        cell_w = self.inner/self.cols
        warp_thick = max(6, int(cell_w*0.9))
        weft_thick = max(6, int(cell_h*0.9))
        area = QRectF(self.margin, self.margin, self.inner, self.inner)
        self.painter.fillRect(area, QColor(225,225,225))
        rows_to_draw = self.rows if self.animate_rows<=0 else min(self.animate_rows, self.rows)
        for r in range(rows_to_draw):
            y = self.margin + r*cell_h + cell_h/2.0
            c = self._weft_color_for_row(r)
            self._draw_weft_ribbon(y, weft_thick, cell_w, warp_thick, r, c)
        for c in range(self.cols):
            x = self.margin + c*cell_w + cell_w/2.0
            col_color = self._warp_color_for_col(c)
            self._draw_warp_ribbon(x, warp_thick, cell_h, weft_thick, c, col_color)
        if self.texture_strength>0.01:
            self._apply_texture()
        if self.show_logic:
            self._draw_logic_overlay(cell_w, cell_h)
        self.painter.end()
        return self.img

    def _draw_weft_ribbon(self, center_y, thickness, cell_w, warp_thick, row_index, color_rgb):
        half = thickness/2.0
        path_left = self.margin - warp_thick
        path_right = self.margin + self.inner + warp_thick
        path_rect = QRectF(path_left, center_y-half, path_right-path_left, thickness)
        grad = QLinearGradient(path_rect.left(), path_rect.top(), path_rect.right(), path_rect.top())
        dark = QColor(max(0,color_rgb[0]-36), max(0,color_rgb[1]-36), max(0,color_rgb[2]-36))
        light = QColor(min(255,color_rgb[0]+36), min(255,color_rgb[1]+36), min(255,color_rgb[2]+36))
        grad.setColorAt(0.0, dark)
        grad.setColorAt(0.5, light)
        grad.setColorAt(1.0, dark)
        self.painter.setPen(Qt.NoPen)
        self.painter.setBrush(QBrush(grad))
        slices = 140
        slice_w = (path_rect.width()) / slices
        amp = max(1.0, thickness*0.12)
        freq = 2.0*math.pi/max(120.0, slices/2.0)
        x0 = path_rect.left()
        for i in range(slices):
            sx = x0 + i*slice_w
            offset = math.sin((i + row_index*3)*freq + row_index) * amp
            sr = QRectF(sx, path_rect.top()+offset, slice_w+1, path_rect.height())
            self.painter.drawRect(sr)
        self.painter.setPen(QPen(QColor(0,0,0,22),1))
        fiber_count = int(max(18, thickness*0.6))
        for f in range(fiber_count):
            fx = path_rect.left() + random.random()*path_rect.width()
            fy = center_y + (random.random()-0.5)*thickness*0.6
            lx = fx + (random.random()-0.5)*thickness*0.6
            ly = fy + (random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy), QPointF(lx,ly))

    def _draw_warp_ribbon(self, center_x, thickness, cell_h, weft_thick, col_index, color_rgb):
        half = thickness/2.0
        path_top = self.margin - weft_thick
        path_bottom = self.margin + self.inner + weft_thick
        path_rect = QRectF(center_x-half, path_top, thickness, path_bottom-path_top)
        grad = QLinearGradient(path_rect.left(), path_rect.top(), path_rect.left(), path_rect.bottom())
        dark = QColor(max(0,color_rgb[0]-36), max(0,color_rgb[1]-36), max(0,color_rgb[2]-36))
        light = QColor(min(255,color_rgb[0]+36), min(255,color_rgb[1]+36), min(255,color_rgb[2]+36))
        grad.setColorAt(0.0,dark)
        grad.setColorAt(0.5,light)
        grad.setColorAt(1.0,dark)
        self.painter.setBrush(QBrush(grad))
        self.painter.setPen(Qt.NoPen)
        slices = 140
        slice_h = path_rect.height()/slices
        amp = max(1.0, thickness*0.12)
        freq = 2.0*math.pi/max(120.0, slices/2.0)
        y0 = path_rect.top()
        for i in range(slices):
            sy = y0 + i*slice_h
            offset = math.sin((i + col_index*2)*freq + col_index) * amp
            sr = QRectF(path_rect.left()+offset, sy, path_rect.width(), slice_h+1)
            self.painter.drawRect(sr)
        self.painter.setPen(QPen(QColor(0,0,0,28),1))
        fiber_count = int(max(18, thickness*0.55))
        for f in range(fiber_count):
            fy = path_rect.top() + random.random()*path_rect.height()
            fx = center_x + (random.random()-0.5)*thickness*0.6
            lx = fx + (random.random()-0.5)*thickness*0.6
            ly = fy + (random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy), QPointF(lx,ly))
        cell_h = self.inner/self.rows
        cell_w = self.inner/self.cols
        half_weft = weft_thick/2.0
        for r in range(self.rows):
            if self.animate_rows>0 and r>=self.animate_rows:
                continue
            bit = self.grid[r][col_index]
            y = self.margin + r*cell_h + cell_h/2.0
            x = center_x
            if bit==0:
                rect = QRectF(x-cell_w/2.0-thickness/2.0, y-half_weft, cell_w+thickness, weft_thick)
                grad_local = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
                grad_local.setColorAt(0.0, QColor(max(0,self.weft_base[0]-30), max(0,self.weft_base[1]-30), max(0,self.weft_base[2]-30)))
                grad_local.setColorAt(0.5, QColor(min(255,self.weft_base[0]+30), min(255,self.weft_base[1]+30), min(255,self.weft_base[2]+30)))
                grad_local.setColorAt(1.0, QColor(max(0,self.weft_base[0]-30), max(0,self.weft_base[1]-30), max(0,self.weft_base[2]-30)))
                self.painter.setBrush(QBrush(grad_local))
                self.painter.setPen(Qt.NoPen)
                self.painter.drawRoundedRect(rect,3,3)

    def _apply_texture(self):
        w = int(self.inner)
        h = int(self.inner)
        seed = self.texture_seed if self.texture_seed is not None else int(time.time()%1e9)
        noise = generate_value_noise(w,h, scale=max(24,int(w/24)), octaves=4, persistence=0.5, seed=seed)
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

class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jacquard Weaver — Full Styles")
        self.resize(1500,920)
        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("Binary punch-card lines. One row per line.")
        self.input_edit.setPlainText("0101010101\n1010101010\n0101010101\n1010101010\n0101010101\n1010101010\n0101010101\n1010101010\n0101010101\n1010101010")
        self.warp_color_btn = QPushButton("Warp color")
        self.weft_color_btn = QPushButton("Weft color")
        self.load_btn = QPushButton("Load .txt")
        self.save_pattern_btn = QPushButton("Save .txt")
        self.weave_btn = QPushButton("Weave (Render)")
        self.save_img_btn = QPushButton("Save Image")
        self.animate_cb = QCheckBox("Animated weaving")
        self.animate_speed_label = QLabel("Speed")
        self.animate_speed = QSpinBox(); self.animate_speed.setRange(10,1000); self.animate_speed.setValue(150)
        self.multicolor_warp_cb = QCheckBox("Multicolor warp")
        self.multicolor_weft_cb = QCheckBox("Multicolor weft")
        self.texture_cb = QCheckBox("Texture overlay")
        self.texture_strength_slider = QSlider(Qt.Horizontal); self.texture_strength_slider.setRange(0,100); self.texture_strength_slider.setValue(20)
        self.show_logic_cb = QCheckBox("Show logic (0/1 overlay)")
        self.palette_style_combo = QComboBox(); self.palette_style_combo.addItems(["Gradient","Rainbow","Monochrome","Warm","Cool"])
        self.canvas_label = QLabel()
        self.canvas_label.setFixedSize(CANVAS_SIZE,CANVAS_SIZE)
        self.layout_ui()
        self.connect_signals()
        self.current_grid = []
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self._animate_step)
        self.anim_row = 0

    def layout_ui(self):
        control_layout = QVBoxLayout()
        control_layout.addWidget(QLabel("Punch-card input"))
        control_layout.addWidget(self.input_edit)
        btn_row1 = QHBoxLayout()
        btn_row1.addWidget(self.warp_color_btn)
        btn_row1.addWidget(self.weft_color_btn)
        btn_row1.addWidget(self.load_btn)
        btn_row1.addWidget(self.save_pattern_btn)
        control_layout.addLayout(btn_row1)
        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self.weave_btn)
        btn_row2.addWidget(self.save_img_btn)
        control_layout.addLayout(btn_row2)
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.animate_cb)
        options_layout.addWidget(self.animate_speed_label)
        options_layout.addWidget(self.animate_speed)
        options_layout.addWidget(self.multicolor_warp_cb)
        options_layout.addWidget(self.multicolor_weft_cb)
        options_layout.addWidget(self.texture_cb)
        options_layout.addWidget(QLabel("Texture %"))
        options_layout.addWidget(self.texture_strength_slider)
        options_layout.addWidget(self.show_logic_cb)
        options_layout.addWidget(QLabel("Palette style"))
        options_layout.addWidget(self.palette_style_combo)
        control_layout.addLayout(options_layout)
        main_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.canvas_label)
        self.setLayout(main_layout)

    def connect_signals(self):
        self.warp_color_btn.clicked.connect(self.choose_warp_color)
        self.weft_color_btn.clicked.connect(self.choose_weft_color)
        self.load_btn.clicked.connect(self.load_txt)
        self.save_pattern_btn.clicked.connect(self.save_txt)
        self.weave_btn.clicked.connect(self.render_weave)
        self.save_img_btn.clicked.connect(self.save_image)

    def choose_warp_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.warp_base = (c.red(), c.green(), c.blue())

    def choose_weft_color(self):
        c = QColorDialog.getColor()
        if c.isValid():
            self.weft_base = (c.red(), c.green(), c.blue())

    def load_txt(self):
        fname,_ = QFileDialog.getOpenFileName(self,"Open Punch Card", "","Text Files (*.txt)")
        if fname:
            with open(fname,"r") as f:
                txt = f.read()
            self.input_edit.setPlainText(txt)

    def save_txt(self):
        fname,_ = QFileDialog.getSaveFileName(self,"Save Punch Card","","Text Files (*.txt)")
        if fname:
            with open(fname,"w") as f:
                f.write(self.input_edit.toPlainText())

    def render_weave(self):
        self.current_grid = parse_input(self.input_edit.toPlainText())
        if not self.current_grid:
            QMessageBox.warning(self,"Error","No valid grid")
            return
        warp_palette, weft_palette = self.generate_palettes()
        canvas = WeaverCanvas(
            self.current_grid,
            warp_base=getattr(self,'warp_base',(0,0,0)),
            weft_base=getattr(self,'weft_base',(255,255,255)),
            multicolor_warp=self.multicolor_warp_cb.isChecked(),
            multicolor_weft=self.multicolor_weft_cb.isChecked(),
            warp_palette=warp_palette,
            weft_palette=weft_palette,
            texture_strength=self.texture_strength_slider.value()/100.0 if self.texture_cb.isChecked() else 0.0,
            show_logic=self.show_logic_cb.isChecked(),
            animate_rows=0
        )
        img = canvas.render()
        self.canvas_label.setPixmap(QPixmap.fromImage(img))

    def generate_palettes(self):
        style = self.palette_style_combo.currentText()
        if style=="Gradient":
            warp_palette = [(random.randint(10,200), random.randint(10,200), random.randint(10,200)),
                            (random.randint(50,255), random.randint(50,255), random.randint(50,255))]
            weft_palette = [(random.randint(10,200), random.randint(10,200), random.randint(10,200)),
                            (random.randint(50,255), random.randint(50,255), random.randint(50,255))]
        elif style=="Rainbow":
            warp_palette = [(255,0,0),(255,127,0),(255,255,0),(0,255,0),(0,0,255),(75,0,130),(148,0,211)]
            weft_palette = [(148,0,211),(75,0,130),(0,0,255),(0,255,0),(255,255,0),(255,127,0),(255,0,0)]
        elif style=="Monochrome":
            base = random.randint(50,200)
            warp_palette = [(base,base,base)]
            weft_palette = [(base+50,base+50,base+50)]
        elif style=="Warm":
            warp_palette = [(random.randint(180,255), random.randint(60,180), random.randint(0,80)),
                            (random.randint(200,255), random.randint(100,200), random.randint(50,150))]
            weft_palette = [(random.randint(180,255), random.randint(60,180), random.randint(0,80)),
                            (random.randint(200,255), random.randint(100,200), random.randint(50,150))]
        elif style=="Cool":
            warp_palette = [(random.randint(0,80), random.randint(100,200), random.randint(180,255)),
                            (random.randint(50,150), random.randint(150,255), random.randint(200,255))]
            weft_palette = [(random.randint(0,80), random.randint(100,200), random.randint(180,255)),
                            (random.randint(50,150), random.randint(150,255), random.randint(200,255))]
        return warp_palette, weft_palette

    def save_image(self):
        fname,_ = QFileDialog.getSaveFileName(self,"Save Image","","PNG Files (*.png);;JPG Files (*.jpg)")
        if fname:
            pixmap = self.canvas_label.pixmap()
            if pixmap:
                pixmap.save(fname)

    def _animate_step(self):
        if not self.current_grid:
            self.timer.stop()
            return
        self.anim_row += 1
        canvas = WeaverCanvas(
            self.current_grid,
            warp_base=getattr(self,'warp_base',(0,0,0)),
            weft_base=getattr(self,'weft_base',(255,255,255)),
            multicolor_warp=self.multicolor_warp_cb.isChecked(),
            multicolor_weft=self.multicolor_weft_cb.isChecked(),
            warp_palette=None,
            weft_palette=None,
            texture_strength=self.texture_strength_slider.value()/100.0 if self.texture_cb.isChecked() else 0.0,
            show_logic=self.show_logic_cb.isChecked(),
            animate_rows=self.anim_row
        )
        img = canvas.render()
        self.canvas_label.setPixmap(QPixmap.fromImage(img))
        if self.anim_row>=len(self.current_grid):
            self.timer.stop()
            self.anim_row=0

if __name__=="__main__":
    app = QApplication(sys.argv)
    w = WeavingApp()
    w.show()
    sys.exit(app.exec_())
