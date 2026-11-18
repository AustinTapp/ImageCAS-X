#!/usr/bin/env python
"""Linked six-panel CT/segmentation reviewer. Run --help for input options."""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import traceback

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from plaque_review_core import (PLANES, ReviewStore, discover_predictions, load_volume,
                               make_demo, read_json, slice_array, source_paths)

pg.setConfigOptions(imageAxisOrder='row-major', antialias=False)
PALETTE = {1:(0, 220, 255), 2:(255, 139, 45), 255:(238, 83, 238)}


class Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class Work(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__(); self.function = function; self.signals = Signals()

    def run(self):
        try:
            self.signals.done.emit(self.function())
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class SliceCanvas(pg.GraphicsLayoutWidget):
    stepped = QtCore.Signal(str, int)
    picked = QtCore.Signal(str, float, float)

    def __init__(self, plane):
        super().__init__(); self.plane = plane
        self.setMinimumHeight(130)
        self.view = self.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.view.setBackgroundColor('#10141c')
        self.ct = pg.ImageItem(); self.seg = pg.ImageItem()
        self.view.addItem(self.ct); self.view.addItem(self.seg)
        self.horizontal = pg.InfiniteLine(angle=0, pen=pg.mkPen('#e6edf3', width=.7))
        self.vertical = pg.InfiniteLine(angle=90, pen=pg.mkPen('#e6edf3', width=.7))
        for line in (self.horizontal, self.vertical):
            line.setZValue(20); self.view.addItem(line, ignoreBounds=True)
        self.scene().sigMouseClicked.connect(self.clicked)
        self.spacing = (1, 1); self.shape = None

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta:
            self.stepped.emit(self.plane, 1 if delta > 0 else -1)
        event.accept()

    def clicked(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.shape:
            p = self.view.mapSceneToView(event.scenePos())
            self.picked.emit(self.plane, p.x()/self.spacing[0] - .5, p.y()/self.spacing[1] - .5)

    def blank(self):
        self.ct.clear(); self.seg.clear(); self.shape = None
        self.horizontal.hide(); self.vertical.hide()

    def display(self, ct, labels, spacing, levels, opacity, visible, classes, crosshair, reset=False):
        self.spacing = spacing; self.shape = ct.shape
        rect = QtCore.QRectF(0, 0, ct.shape[1]*spacing[0], ct.shape[0]*spacing[1])
        self.ct.setImage(np.ascontiguousarray(ct), autoLevels=False, levels=levels)
        self.ct.setRect(rect)
        colors = np.zeros((*labels.shape, 4), np.uint8)
        for value in classes:
            selected = labels == value
            colors[selected, :3] = PALETTE[value]
            colors[selected, 3] = int(255*opacity)
        self.seg.setImage(colors, autoLevels=False); self.seg.setRect(rect); self.seg.setVisible(visible)
        self.vertical.setPos((crosshair[0]+.5)*spacing[0]); self.horizontal.setPos((crosshair[1]+.5)*spacing[1])
        self.horizontal.show(); self.vertical.show()
        if reset:
            self.view.setRange(rect, padding=.015)


class Reviewer(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__(); self.args = args
        self.manifest = read_json(args.dataset)
        self.cases = sorted(case for case, row in self.manifest['cases'].items() if row['partition'] == args.partition)
        if not self.cases:
            raise ValueError(f'No cases in {args.partition}')
        self.sets = discover_predictions(args.bench, self.manifest, args.partition, args.runs, args.prediction)
        self.store = ReviewStore(args.out, args.references_only)
        self.records = self.store.records()
        self.volume = None; self.shared = None; self.indices = [0, 0, 0]
        self.token = 0; self.exporting = False; self.loading = False; self.tasks = set()
        self.case = self.cases[0]; self.predictions = self.sets[0]; self.last_plane = 'Axial'
        self.pool = QtCore.QThreadPool(self); self.pool.setMaxThreadCount(2)
        self.build_ui(); self.refresh_table(); self.load_case()
        QtWidgets.QApplication.instance().installEventFilter(self)
        self.setWindowTitle('PlaquePilot • Case review')
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1550,screen.width()-40),min(1040,screen.height()-60))

    def build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central); outer.setContentsMargins(12, 10, 12, 8)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('PlaquePilot  /  CASE REVIEW'); title.setStyleSheet('font-size:19px;font-weight:600;')
        top.addWidget(title); top.addStretch()
        top.addWidget(QtWidgets.QLabel('Prediction:'))
        self.model = QtWidgets.QComboBox()
        for item in self.sets: self.model.addItem(item.name)
        self.model.setMinimumWidth(390); self.model.currentIndexChanged.connect(self.change_model); top.addWidget(self.model)
        outer.addLayout(top)
        self.info = QtWidgets.QLabel(); self.info.setWordWrap(True); outer.addWidget(self.info)
        self.notice = QtWidgets.QLabel(); self.notice.setWordWrap(True); self.notice.setStyleSheet('color:#ffcd7a;'); outer.addWidget(self.notice)
        split = QtWidgets.QSplitter(); outer.addWidget(split, 1)
        sidebar = QtWidgets.QWidget(); side = QtWidgets.QVBoxLayout(sidebar); side.setContentsMargins(0, 0, 8, 0)
        self.search = QtWidgets.QLineEdit(); self.search.setPlaceholderText('Filter case ID…'); self.search.textChanged.connect(self.filter_cases)
        side.addWidget(self.search)
        self.table = QtWidgets.QTableWidget(0, 3); self.table.setHorizontalHeaderLabels(['Case', 'Plaque Dice', 'Review'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide(); self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellClicked.connect(self.select_row); self.table.setMinimumWidth(310); side.addWidget(self.table, 1)
        order = QtWidgets.QPushButton('Sort by plaque Dice ↑'); order.clicked.connect(self.sort_cases); side.addWidget(order)
        self.count = QtWidgets.QLabel(); side.addWidget(self.count)
        hint = QtWidgets.QLabel('Cyan: lumen  •  Orange: plaque\nMagenta: ignored / unknown (255)\n\nWheel: linked slice scroll\nLeft click: linked crosshair\nCtrl + wheel: zoom\nRight drag: zoom  •  Middle drag: pan\n\nA: approve  •  R: reject  •  F: review\nU: reset to unreviewed\nN / →: next  •  B / ←: previous\nG: ground truth  •  P: prediction\nSpace: toggle both overlays\nJ: jump to plaque  •  Home: fit views')
        hint.setWordWrap(True); hint.setStyleSheet('color:#aebccc;'); side.addWidget(hint)
        split.addWidget(sidebar)
        display = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(display); grid.setContentsMargins(4, 0, 0, 0)
        self.gt_toggle = QtWidgets.QCheckBox('GROUND TRUTH   [G]'); self.gt_toggle.setChecked(True)
        self.pred_toggle = QtWidgets.QCheckBox('PREDICTION   [P]'); self.pred_toggle.setChecked(True)
        for col, control in enumerate((self.gt_toggle, self.pred_toggle)):
            control.setStyleSheet('font-size:15px;font-weight:600;'); control.toggled.connect(self.render)
            grid.addWidget(control, 0, col)
        self.canvases = {}; self.sliders = {}; self.slice_labels = {}
        edges = {'Axial':'R  ←  AXIAL  →  L    |    A top · P bottom',
                 'Coronal':'R  ←  CORONAL  →  L    |    S top · I bottom',
                 'Sagittal':'P  ←  SAGITTAL  →  A    |    S top · I bottom'}
        for row, plane in enumerate(PLANES):
            for col in (0, 1):
                panel = QtWidgets.QWidget(); box = QtWidgets.QVBoxLayout(panel); box.setContentsMargins(0, 0, 0, 0); box.setSpacing(2)
                text = QtWidgets.QLabel(edges[plane]); text.setStyleSheet('color:#9bacbf;font-size:10px;'); box.addWidget(text)
                canvas = SliceCanvas(plane); canvas.stepped.connect(self.step); canvas.picked.connect(self.pick)
                box.addWidget(canvas, 1); self.canvases[plane, col] = canvas
                grid.addWidget(panel, row*2+1, col)
            self.canvases[plane, 1].view.setXLink(self.canvases[plane, 0].view)
            self.canvases[plane, 1].view.setYLink(self.canvases[plane, 0].view)
            bar = QtWidgets.QWidget(); bar_layout = QtWidgets.QHBoxLayout(bar); bar_layout.setContentsMargins(0, 0, 0, 2)
            label = QtWidgets.QLabel(plane); label.setMinimumWidth(220); bar_layout.addWidget(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.valueChanged.connect(lambda value, p=plane:self.set_slice(p, value))
            bar_layout.addWidget(slider); self.sliders[plane] = slider; self.slice_labels[plane] = label
            grid.addWidget(bar, row*2+2, 0, 1, 2); grid.setRowStretch(row*2+1, 1)
        split.addWidget(display); split.setSizes([330, 1200])
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('Window center / width (HU):'))
        self.level = QtWidgets.QSpinBox(); self.level.setRange(-1500, 3000); self.level.setValue(250); self.level.valueChanged.connect(self.render)
        self.width = QtWidgets.QSpinBox(); self.width.setRange(1, 6000); self.width.setValue(800); self.width.valueChanged.connect(self.render)
        controls.addWidget(self.level); controls.addWidget(self.width)
        controls.addWidget(QtWidgets.QLabel('Opacity:'))
        self.opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.opacity.setRange(0, 100); self.opacity.setValue(42); self.opacity.setMaximumWidth(140)
        self.opacity.valueChanged.connect(self.render); controls.addWidget(self.opacity)
        self.classes = {}
        for value, name in ((1, 'Lumen'), (2, 'Plaque'), (255, 'Ignored')):
            control = QtWidgets.QCheckBox(name); control.setChecked(True); control.toggled.connect(self.render)
            controls.addWidget(control); self.classes[value] = control
        self.resample = QtWidgets.QCheckBox('Resample mismatched labels for display'); self.resample.setChecked(self.args.resample_labels)
        self.resample.toggled.connect(self.reload_case); controls.addWidget(self.resample)
        outer.addLayout(controls)
        notes = QtWidgets.QHBoxLayout()
        self.reason = QtWidgets.QComboBox(); self.reason.addItems(['', 'False positives', 'Missed plaque', 'Lumen error', 'Boundary error', 'Geometry / alignment', 'Ground truth concern', 'Other'])
        self.notes = QtWidgets.QLineEdit(); self.notes.setPlaceholderText('Review notes (saved with decision; source labels stay unchanged)')
        notes.addWidget(QtWidgets.QLabel('Reason:')); notes.addWidget(self.reason); notes.addWidget(self.notes, 1); outer.addLayout(notes)
        actions = QtWidgets.QHBoxLayout(); self.action_buttons = []
        for label, category, color in [('Approve [A]', 'approved', '#23694e'), ('Reject [R]', 'rejected', '#84383c'), ('Further review [F]', 'needs_review', '#816322'), ('Reset [U]', 'unreviewed', '#3d4a5c')]:
            button = QtWidgets.QPushButton(label); button.setStyleSheet(f'background:{color};padding:8px 18px;')
            button.clicked.connect(lambda checked=False, c=category:self.decide(c)); actions.addWidget(button); self.action_buttons.append(button)
        self.auto_next = QtWidgets.QCheckBox('Advance after decision'); self.auto_next.setChecked(True); actions.addWidget(self.auto_next)
        actions.addStretch()
        previous = QtWidgets.QPushButton('← Previous'); previous.clicked.connect(lambda:self.move_case(-1)); actions.addWidget(previous)
        next_button = QtWidgets.QPushButton('Next →'); next_button.clicked.connect(lambda:self.move_case(1)); actions.addWidget(next_button)
        outer.addLayout(actions)
        self.statusBar().showMessage('All decisions apply to a case + prediction set. Original CTs, labels and predictions are preserved.')
        self.setStyleSheet('QWidget{background:#171e29;color:#e1e9f3;font-size:12px;} QLineEdit,QSpinBox,QComboBox,QTableWidget{background:#202a38;} QPushButton{border:1px solid #46576b;border-radius:4px;padding:5px;} QHeaderView::section{background:#263244;padding:5px;} QTableWidget::item:selected{background:#315070;}')

    def run_task(self, function, done, failed):
        job = Work(function); self.tasks.add(job)
        def success(value):
            self.tasks.discard(job); done(value)
        def failure(value):
            self.tasks.discard(job); failed(value)
        job.signals.done.connect(success); job.signals.failed.connect(failure); self.pool.start(job)

    def refresh_table(self):
        self.table.setRowCount(len(self.cases))
        for i, case in enumerate(self.cases):
            metric = self.predictions.metrics.get(case, {}).get('plaque', {}).get('dice')
            status = self.records.get(self.store.key(self.predictions.name, case), {}).get('category', 'unreviewed')
            if not (self.predictions.root/f'{case}.nii.gz').is_file(): status += ' / MISSING'
            for col, value in enumerate((case, f'{float(metric):.3f}' if metric not in ('', None) else '—', status)):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(value))
        self.table.resizeColumnsToContents(); self.filter_cases()
        self.count.setText(f'{len(self.cases)} cases · {sum(self.store.key(self.predictions.name,c) in self.records and self.records[self.store.key(self.predictions.name,c)]["category"] != "unreviewed" for c in self.cases)} reviewed')
        if self.case in self.cases: self.table.selectRow(self.cases.index(self.case))

    def filter_cases(self):
        query = self.search.text().strip().lower()
        for i, case in enumerate(self.cases): self.table.setRowHidden(i, query not in case.lower())

    def sort_cases(self):
        def score(case):
            value = self.predictions.metrics.get(case, {}).get('plaque', {}).get('dice')
            return float(value) if value not in ('', None) else float('inf')
        self.cases.sort(key=lambda case:(score(case), case)); self.refresh_table()

    def select_row(self, row, column):
        if self.exporting: return
        self.case = self.cases[row]; self.load_case()

    def change_model(self, index):
        if self.exporting: return
        self.predictions = self.sets[index]; self.refresh_table(); self.load_case(preserve_position=True)

    def move_case(self, delta):
        if self.exporting: return
        visible = [case for i,case in enumerate(self.cases) if not self.table.isRowHidden(i)]
        if not visible: return
        index = visible.index(self.case) if self.case in visible else 0
        new_index = min(max(index+delta, 0), len(visible)-1)
        if new_index == index:
            self.statusBar().showMessage('Reached the end of the visible case list.'); return
        self.case = visible[new_index]; self.table.selectRow(self.cases.index(self.case)); self.load_case()

    def reload_case(self):
        self.shared = None; self.load_case()

    def load_case(self, preserve_position=False):
        self.token += 1; token = self.token
        self.loading = True; self.volume = None
        for canvas in self.canvases.values(): canvas.blank()
        self.info.setText(f'{self.case}  •  Loading CT / labels…')
        self.notice.setText('')
        record = self.records.get(self.store.key(self.predictions.name, self.case), {})
        self.notes.setText(record.get('notes', '')); self.reason.setCurrentText(record.get('reason', ''))
        paths = source_paths(self.args.dataset, self.manifest['cases'][self.case], self.predictions, self.case)
        case, task, resample = self.case, self.predictions.task, self.resample.isChecked()
        shared = self.shared if preserve_position else None
        position = list(self.indices) if shared and shared.case == self.case else None
        self.run_task(lambda:load_volume(case, paths, task, resample, shared),
                      lambda value:self.loaded(token, value, position), lambda error:self.load_failed(token,error))

    def loaded(self, token, volume, position):
        if token != self.token: return
        self.volume = volume; self.shared = volume.shared; self.loading = False
        if position is None:
            mask = self.shared.gt == 2
            if not mask.any(): mask = self.shared.gt == 1
            # Center on the largest connected annotation component, not its bounding-box midpoint.
            if mask.any():
                from scipy import ndimage
                labels, n = ndimage.label(mask)
                sizes = np.bincount(labels.ravel()); sizes[0] = 0
                position = [int(round(v)) for v in ndimage.center_of_mass(mask, labels, int(sizes.argmax()))]
            else: position = [v//2 for v in self.shared.ct.shape]
        self.indices = position
        for plane, (axis, *_) in PLANES.items():
            slider = self.sliders[plane]; slider.blockSignals(True); slider.setRange(0, self.shared.ct.shape[axis]-1); slider.setValue(self.indices[axis]); slider.blockSignals(False)
        self.notice.setText('  '.join(volume.warnings))
        self.render(reset=True)

    def load_failed(self, token, error):
        if token != self.token: return
        self.loading = False; self.shared = None; self.volume = None
        message = error.strip().splitlines()[-1]
        self.info.setText(f'{self.case}  •  Cannot display this case')
        self.notice.setText(message + '   You can flag/reject it and record a reason; no case is silently skipped.')
        self.statusBar().showMessage('Load failed. See the terminal for details.'); print(error, file=sys.stderr)

    def set_slice(self, plane, value):
        if not self.volume: return
        self.last_plane = plane; self.indices[PLANES[plane][0]] = int(value); self.render()

    def step(self, plane, delta):
        self.last_plane = plane
        self.sliders[plane].setValue(self.sliders[plane].value()+delta)

    def pick(self, plane, x, y):
        if not self.volume: return
        from plaque_review_core import screen_to_indices
        self.indices = screen_to_indices(x, y, self.shared.ct.shape, self.indices, plane)
        for name, (axis, *_) in PLANES.items():
            slider = self.sliders[name]; slider.blockSignals(True); slider.setValue(self.indices[axis]); slider.blockSignals(False)
        self.last_plane = plane; self.render()

    def jump_to_plaque(self):
        if not self.volume: return
        mask = self.shared.gt == 2
        if not mask.any(): mask = self.volume.pred == 2
        coords = np.argwhere(mask)
        if len(coords):
            # Choose an actual plaque voxel near the center, even for disjoint lesions.
            center = coords.mean(axis=0); self.indices = coords[np.argmin(((coords-center)**2).sum(axis=1))].tolist()
            for name, (axis, *_) in PLANES.items():
                slider = self.sliders[name]; slider.blockSignals(True); slider.setValue(self.indices[axis]); slider.blockSignals(False)
            self.render()

    def render(self, *unused, reset=False):
        if self.volume is None: return
        width = self.width.value(); center = self.level.value(); levels = (center-width/2, center+width/2)
        spacing = self.shared.reference.header.get_zooms()[:3]
        classes = [value for value, control in self.classes.items() if control.isChecked()]
        for plane, (axis, horizontal, vertical, flip_x) in PLANES.items():
            shape = self.shared.ct.shape
            crosshair = (shape[horizontal]-1-self.indices[horizontal] if flip_x else self.indices[horizontal], shape[vertical]-1-self.indices[vertical])
            ct = slice_array(self.shared.ct, self.indices, plane)
            for col, label in ((0, self.shared.gt), (1, self.volume.pred)):
                self.canvases[plane, col].display(ct, slice_array(label,self.indices,plane), (spacing[horizontal],spacing[vertical]), levels,
                    self.opacity.value()/100, (self.gt_toggle if col == 0 else self.pred_toggle).isChecked(), classes, crosshair, reset)
            self.slice_labels[plane].setText(f'{plane}  {self.indices[axis]+1} / {shape[axis]}')
        def number(value): return 'undefined' if value is None or value == '' else f'{float(value):.3f}'
        recorded = self.predictions.metrics.get(self.case, {})
        metrics = recorded or self.volume.metrics
        metric_origin = 'recorded full-volume metrics' if recorded else 'display-grid overlap'
        scores = '   |   '.join(f'{name}: Dice {number(row.get("dice"))} · P {number(row.get("precision"))} · R {number(row.get("recall"))}' for name,row in metrics.items())
        world = self.shared.reference.affine @ np.array([*self.indices,1])
        status = self.records.get(self.store.key(self.predictions.name,self.case),{}).get('category','unreviewed')
        self.info.setText(f'{self.case}  •  {status}  •  RAS {world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f} mm  •  {metric_origin}  •  {scores}')

    def toggle_both(self):
        value = not (self.gt_toggle.isChecked() or self.pred_toggle.isChecked())
        self.gt_toggle.setChecked(value); self.pred_toggle.setChecked(value)

    def decide(self, category):
        if self.exporting or self.loading: return
        if category == 'approved' and self.volume is None:
            self.statusBar().showMessage('A case must load successfully before it can be approved. Use Further review for missing/unreadable files.'); return
        self.exporting = True
        self.notes.clearFocus(); self.reason.clearFocus()
        for button in self.action_buttons: button.setEnabled(False)
        self.model.setEnabled(False); self.table.setEnabled(False); self.resample.setEnabled(False)
        paths = source_paths(self.args.dataset,self.manifest['cases'][self.case],self.predictions,self.case)
        model, case = self.predictions.name, self.case
        notes, reason = self.notes.text(), self.reason.currentText()
        view = dict(indices_ras=list(self.indices) if self.volume else None, window_center_hu=self.level.value(),window_width_hu=self.width.value(),
                    display_resampling=self.resample.isChecked(),warnings=self.volume.warnings if self.volume else ['Case failed to load'])
        self.statusBar().showMessage(f'Saving {case} → {category} (first decision copies source files into the review collection)…')
        self.run_task(lambda:self.store.decide(model,case,category,paths,notes=notes,reason=reason,view=view,reviewer=self.args.reviewer), self.saved, self.save_failed)

    def export_finished(self):
        self.exporting = False
        for button in self.action_buttons: button.setEnabled(True)
        self.model.setEnabled(True); self.table.setEnabled(True); self.resample.setEnabled(True)

    def saved(self, record):
        self.export_finished(); self.records[self.store.key(record['model'],record['case'])] = record
        self.refresh_table(); self.render()
        self.statusBar().showMessage(f'Saved {record["case"]}: {record["category"]} → {self.args.out / record["folder"]}')
        if record.get('export_warnings'):
            self.statusBar().showMessage(' | '.join(record['export_warnings']))
        if self.auto_next.isChecked(): self.move_case(1)

    def save_failed(self, error):
        self.export_finished(); print(error, file=sys.stderr)
        QtWidgets.QMessageBox.critical(self,'Review was not saved',error.strip().splitlines()[-1]+'\nPrevious review remains available. See terminal for details.')

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.KeyPress and not QtWidgets.QApplication.activePopupWidget():
            if self.handle_hotkey(event): return True
        return super().eventFilter(watched, event)

    def handle_hotkey(self, event):
        if event.modifiers() & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.AltModifier | QtCore.Qt.KeyboardModifier.MetaModifier):
            return False
        focus = QtWidgets.QApplication.focusWidget()
        if isinstance(focus, (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
            return False
        key = event.key(); K = QtCore.Qt.Key
        actions = {K.Key_A:lambda:self.decide('approved'),K.Key_R:lambda:self.decide('rejected'),K.Key_F:lambda:self.decide('needs_review'),
                   K.Key_U:lambda:self.decide('unreviewed'),K.Key_N:lambda:self.move_case(1),K.Key_Right:lambda:self.move_case(1),
                   K.Key_B:lambda:self.move_case(-1),K.Key_Left:lambda:self.move_case(-1),K.Key_G:self.gt_toggle.toggle,
                   K.Key_P:self.pred_toggle.toggle,K.Key_Space:self.toggle_both,K.Key_J:self.jump_to_plaque,
                   K.Key_Home:lambda:self.render(reset=True),K.Key_Up:lambda:self.step(self.last_plane,1),K.Key_Down:lambda:self.step(self.last_plane,-1)}
        if key in actions:
            if not event.isAutoRepeat() or key in (K.Key_Up, K.Key_Down): actions[key]()
            event.accept(); return True
        return False

    def closeEvent(self,event):
        if self.exporting or self.loading or self.tasks:
            self.statusBar().showMessage('Wait for loading/export to finish before closing.'); event.ignore()
        else: event.accept()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench',type=Path,default=Path('experiments/PlaquePilot_confirmed'))
    p.add_argument('--dataset',type=Path,help='Default: BENCH/prepared/dataset.json')
    p.add_argument('--runs',nargs='+',help='Names beneath BENCH/runner; default: the two runs used in the five-model comparison')
    p.add_argument('--prediction',action='append',help='Explicit prediction set NAME=DIRECTORY; repeat for multiple models')
    p.add_argument('--partition',choices=('train','val','test'),default='val')
    p.add_argument('--out',type=Path,default=Path('experiments/PlaquePilot_case_review'))
    p.add_argument('--resample-labels',action='store_true',help='Explicit nearest-neighbor physical resampling for display; never changes source files')
    p.add_argument('--references-only',action='store_true',help='Save review metadata/folders without copying NIfTI files')
    p.add_argument('--reviewer',default=os.environ.get('USERNAME',os.environ.get('USER','')))
    p.add_argument('--demo',action='store_true',help='Generate two tiny synthetic cases; no patient data needed')
    return p


def main():
    args = parser().parse_args()
    if args.demo:
        args.dataset,args.prediction = make_demo(args.out/'synthetic_demo')
    args.dataset = args.dataset or args.bench/'prepared/dataset.json'
    app = QtWidgets.QApplication(sys.argv[:1]); app.setStyle('Fusion')
    try: window = Reviewer(args)
    except Exception as exc:
        traceback.print_exc(); QtWidgets.QMessageBox.critical(None,'Cannot open reviewer',str(exc)); return 1
    window.show(); return app.exec()


if __name__ == '__main__':
    sys.exit(main())
