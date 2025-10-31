# weaving_full_resize.py
# Jacquard Punch-Card Weaver — full version with runtime canvas size
# Dependencies: PyQt5, Pillow (optional)
# Usage: python weaving_full_resize.py

import sys, math, random, time
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QColor, QLinearGradient, QPixmap, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QColorDialog, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QCheckBox, QSlider, QSpinBox, QComboBox
)

# ------------------ Helpers ------------------

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
        v00 = base[iy][ix]; v10 = base[iy][ix+1]
        v01 = base[iy+1][ix]; v11 = base[iy+1][ix+1]
        a = lerp(v00, v10, fx_s)
        b = lerp(v01, v11, fx_s)
        return lerp(a, b, fy_s)
    img = [[0.0]*width for _ in range(height)]
    amplitude = 1.0; freq_scale = 1.0; max_amp = 0.0
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

# ------------------ WeaverCanvas ------------------

class WeaverCanvas:
    def __init__(self, grid, warp_base=(0,0,0), weft_base=(255,255,255),
                 canvas_width=1200, canvas_height=1200, multicolor_warp=False, multicolor_weft=False,
                 warp_palette=None, weft_palette=None,
                 texture_strength=0.25, texture_seed=None,
                 show_logic=False, animate_rows=0):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.warp_base = warp_base
        self.weft_base = weft_base
        self.width = canvas_width
        self.height = canvas_height
        self.multicolor_warp = multicolor_warp
        self.multicolor_weft = multicolor_weft
        self.warp_palette = warp_palette or [warp_base]
        self.weft_palette = weft_palette or [weft_base]
        self.texture_strength = texture_strength
        self.texture_seed = texture_seed
        self.show_logic = show_logic
        self.animate_rows = animate_rows
        self.img = QImage(self.width, self.height, QImage.Format_RGB32)
        self.img.fill(QColor(240,240,240))
        self.painter = QPainter(self.img)
        self.painter.setRenderHint(QPainter.Antialiasing)
        self.margin = int(min(self.width,self.height)*0.03)
        self.inner_w = self.width - 2*self.margin
        self.inner_h = self.height - 2*self.margin

    def _warp_color_for_col(self, col_idx):
        if not self.multicolor_warp or not self.warp_palette:
            return self.warp_base
        t = col_idx / max(1, self.cols-1)
        n = len(self.warp_palette)
        if n == 1: return self.warp_palette[0]
        seg = t*(n-1); i=int(seg); frac=seg-i
        return lerp_color(self.warp_palette[i], self.warp_palette[min(i+1,n-1)], frac)

    def _weft_color_for_row(self, row_idx):
        if not self.multicolor_weft or not self.weft_palette:
            return self.weft_base
        t = row_idx / max(1,self.rows-1)
        n=len(self.weft_palette)
        if n==1: return self.weft_palette[0]
        seg=t*(n-1); i=int(seg); frac=seg-i
        return lerp_color(self.weft_palette[i], self.weft_palette[min(i+1,n-1)], frac)

    def render(self):
        if self.rows==0 or self.cols==0:
            self.painter.fillRect(0,0,self.width,self.height,QColor(240,240,240))
            self.painter.end(); return self.img

        cell_w = self.inner_w / self.cols
        cell_h = self.inner_h / self.rows
        warp_thick = max(6,int(cell_w*0.9))
        weft_thick = max(6,int(cell_h*0.9))

        self.painter.fillRect(self.margin,self.margin,self.inner_w,self.inner_h,QColor(225,225,225))

        rows_to_draw = self.rows if self.animate_rows<=0 else min(self.animate_rows,self.rows)
        for r in range(rows_to_draw):
            y = self.margin + r*cell_h + cell_h/2.0
            c = self._weft_color_for_row(r)
            self._draw_weft_ribbon(y,weft_thick,cell_w,warp_thick,r,c)

        for c in range(self.cols):
            x = self.margin + c*cell_w + cell_w/2.0
            col_color = self._warp_color_for_col(c)
            self._draw_warp_ribbon(x,warp_thick,cell_h,weft_thick,c,col_color)

        if self.texture_strength>0.01: self._apply_texture()
        if self.show_logic: self._draw_logic_overlay(cell_w,cell_h)

        self.painter.end()
        return self.img

    # ------------------ Drawing methods ------------------

    def _draw_weft_ribbon(self, center_y, thickness, cell_w, warp_thick, row_index, color_rgb):
        half=thickness/2.0
        path_rect = QRectF(self.margin-warp_thick, center_y-half, self.inner_w+2*warp_thick, thickness)
        grad = QLinearGradient(path_rect.left(),path_rect.top(),path_rect.right(),path_rect.top())
        base=QColor(*color_rgb)
        dark=QColor(max(0,color_rgb[0]-36),max(0,color_rgb[1]-36),max(0,color_rgb[2]-36))
        light=QColor(min(255,color_rgb[0]+36),min(255,color_rgb[1]+36),min(255,color_rgb[2]+36))
        grad.setColorAt(0.0,dark); grad.setColorAt(0.5,light); grad.setColorAt(1.0,dark)
        self.painter.setBrush(QBrush(grad)); self.painter.setPen(Qt.NoPen)
        slices=140; slice_w=path_rect.width()/slices
        amp=max(1.0,thickness*0.12); freq=2*math.pi/max(120.0,slices/2.0)
        x0=path_rect.left()
        for i in range(slices):
            sx=x0+i*slice_w
            offset=math.sin((i+row_index*3)*freq+row_index)*amp
            self.painter.drawRect(QRectF(sx,path_rect.top()+offset,slice_w+1,path_rect.height()))
        self.painter.setPen(QPen(QColor(0,0,0,22),1))
        for f in range(int(max(18,thickness*0.6))):
            fx=path_rect.left()+random.random()*path_rect.width()
            fy=center_y+(random.random()-0.5)*thickness*0.6
            lx=fx+(random.random()-0.5)*thickness*0.6
            ly=fy+(random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy),QPointF(lx,ly))

    def _draw_warp_ribbon(self, center_x, thickness, cell_h, weft_thick, col_index, color_rgb):
        half=thickness/2.0
        path_rect=QRectF(center_x-half,self.margin-weft_thick,thickness,self.inner_h+2*weft_thick)
        grad=QLinearGradient(path_rect.left(),path_rect.top(),path_rect.left(),path_rect.bottom())
        dark=QColor(max(0,color_rgb[0]-36),max(0,color_rgb[1]-36),max(0,color_rgb[2]-36))
        light=QColor(min(255,color_rgb[0]+36),min(255,color_rgb[1]+36),min(255,color_rgb[2]+36))
        grad.setColorAt(0.0,dark); grad.setColorAt(0.5,light); grad.setColorAt(1.0,dark)
        self.painter.setBrush(QBrush(grad)); self.painter.setPen(Qt.NoPen)
        slices=140; slice_h=path_rect.height()/slices; amp=max(1.0,thickness*0.12); freq=2*math.pi/max(120.0,slices/2.0)
        y0=path_rect.top()
        for i in range(slices):
            sy=y0+i*slice_h
            offset=math.sin((i+col_index*2)*freq+col_index)*amp
            self.painter.drawRect(QRectF(path_rect.left()+offset,sy,path_rect.width(),slice_h+1))
        self.painter.setPen(QPen(QColor(0,0,0,28),1))
        for f in range(int(max(18,thickness*0.55))):
            fy=path_rect.top()+random.random()*path_rect.height()
            fx=center_x+(random.random()-0.5)*thickness*0.6
            lx=fx+(random.random()-0.5)*thickness*0.6
            ly=fy+(random.random()-0.5)*thickness*0.6
            self.painter.drawLine(QPointF(fx,fy),QPointF(lx,ly))
        cell_w=self.inner_w/self.cols; cell_h=self.inner_h/self.rows; half_weft=weft_thick/2.0
        for r in range(self.rows):
            if self.animate_rows>0 and r>=self.animate_rows: continue
            bit=self.grid[r][col_index]; y=self.margin+r*cell_h+cell_h/2.0; x=center_x
            if bit==0:
                rect=QRectF(x-cell_w/2-thickness/2,y-half_weft,cell_w+thickness,weft_thick)
                grad_local=QLinearGradient(rect.left(),rect.top(),rect.right(),rect.top())
                grad_local.setColorAt(0.0,QColor(max(0,self.weft_base[0]-30),max(0,self.weft_base[1]-30),max(0,self.weft_base[2]-30)))
                grad_local.setColorAt(0.5,QColor(min(255,self.weft_base[0]+30),min(255,self.weft_base[1]+30),min(255,self.weft_base[2]+30)))
                grad_local.setColorAt(1.0,QColor(max(0,self.weft_base[0]-30),max(0,self.weft_base[1]-30),max(0,self.weft_base[2]-30)))
                self.painter.setBrush(QBrush(grad_local)); self.painter.setPen(Qt.NoPen)
                self.painter.drawRoundedRect(rect,3,3)

    def _apply_texture(self):
        w=int(self.inner_w); h=int(self.inner_h)
        seed=self.texture_seed if self.texture_seed is not None else int(time.time()%1e9)
        noise=generate_value_noise(w,h,scale=max(24,int(w/24)),octaves=4,persistence=0.5,seed=seed)
        for yy in range(h):
            for xx in range(w):
                global_x=int(self.margin+xx); global_y=int(self.margin+yy)
                base_col=QColor(self.img.pixel(global_x,global_y))
                n=noise[yy][xx]/255.0
                s=self.texture_strength
                if n<0.5: t=(0.5-n)*2*s; new_r=int(base_col.red()*(1-t)); new_g=int(base_col.green()*(1-t)); new_b=int(base_col.blue()*(1-t))
                else: t=(n-0.5)*2*s; new_r=int(base_col.red()+(255-base_col.red())*t*0.25); new_g=int(base_col.green()+(255-base_col.green())*t*0.25); new_b=int(base_col.blue()+(255-base_col.blue())*t*0.25)
                self.img.setPixel(global_x,global_y,QColor(new_r,new_g,new_b).rgb())

    def _draw_logic_overlay(self, cell_w, cell_h):
        pen=QPen(QColor(0,0,0,80),1)
        self.painter.setPen(pen)
        for c in range(self.cols+1): self.painter.drawLine(QPointF(self.margin+c*cell_w,self.margin),QPointF(self.margin+c*cell_w,self.margin+self.inner_h))
        for r in range(self.rows+1): self.painter.drawLine(QPointF(self.margin,self.margin+r*cell_h),QPointF(self.margin+self.inner_w,self.margin+r*cell_h))
        font=self.painter.font(); font.setPointSize(max(8,int(min(cell_w,cell_h)*0.35))); self.painter.setFont(font)
        for r in range(self.rows):
            for c in range(self.cols):
                bit=self.grid[r][c]; x=self.margin+c*cell_w+cell_w/2; y=self.margin+r*cell_h+cell_h/2
                self.painter.setPen(QPen(QColor(255,255,255,180) if bit else QColor(0,0,0,160)))
                self.painter.drawText(QRectF(x-15,y-10,30,20),Qt.AlignCenter,str(bit))

# ------------------ GUI ------------------

class WeavingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jacquard Weaving Simulator — Full Version")
        self.resize(1280, 960)
        self.grid_input=QTextEdit("01010\n10101\n01010\n10101")
        self.render_btn=QPushButton("Render")
        self.render_btn.clicked.connect(self.render_grid)
        self.image_label=QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.canvas_width_spin=QSpinBox(); self.canvas_width_spin.setRange(100,4000); self.canvas_width_spin.setValue(1200); self.canvas_width_spin.setPrefix("W: ")
        self.canvas_height_spin=QSpinBox(); self.canvas_height_spin.setRange(100,4000); self.canvas_height_spin.setValue(1200); self.canvas_height_spin.setPrefix("H: ")
        layout=QVBoxLayout()
        hlayout=QHBoxLayout(); hlayout.addWidget(self.canvas_width_spin); hlayout.addWidget(self.canvas_height_spin); hlayout.addWidget(self.render_btn)
        layout.addWidget(self.grid_input); layout.addLayout(hlayout); layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.render_grid()

    def render_grid(self):
        text=self.grid_input.toPlainText()
        grid=parse_input(text)
        if not grid:
            QMessageBox.warning(self,"Error","Grid is empty or invalid!")
            return
        width=self.canvas_width_spin.value()
        height=self.canvas_height_spin.value()
        canvas=WeaverCanvas(grid,canvas_width=width,canvas_height=height)
        img=canvas.render()
        self.image_label.setPixmap(QPixmap.fromImage(img).scaled(self.image_label.width(),self.image_label.height(),Qt.KeepAspectRatio))

# ------------------ Run ------------------

if __name__=="__main__":
    app=QApplication(sys.argv)
    w=WeavingApp()
    w.show()
    sys.exit(app.exec_())
