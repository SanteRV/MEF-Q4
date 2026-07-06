"""Widget para animar la aplicación progresiva de la carga sobre la deformada.

Permite al usuario "ver" cómo se deforma la estructura desde el estado sin carga
(0%) hasta la deformación total (100%) mediante un slider y controles de
reproducción. Es puramente un widget de UI: no calcula nada, solo emite la
señal `progress_changed(percent: float)` para que MainWindow re-renderice la
deformada escalando los desplazamientos por `percent/100`.

Uso típico:
    animator = LoadAnimator()
    animator.progress_changed.connect(on_progress_changed)
    layout.addWidget(animator)
    animator.play()
"""
from __future__ import annotations

import qtawesome as qta
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from .theme import Colors


# --- Constantes de animación ----------------------------------------------
_TIMER_INTERVAL_MS = 33      # ~30 FPS
_STEP_PER_TICK = 2           # incremento del slider por tick (0..100)
_BUTTON_SIZE = 40            # botones cuadrados 40x40
_WIDGET_MAX_HEIGHT = 44      # margen pequeño sobre los botones


class LoadAnimator(QWidget):
    """Widget con controles para animar la aplicación de carga 0% -> 100%.

    Emite la señal ``progress_changed(percent: float)`` cuando el slider
    cambia (sea por interacción manual, por avance del timer o por reset),
    así MainWindow puede re-renderizar la deformada con
    ``displacements * percent / 100``.
    """

    progress_changed = Signal(float)  # % de 0 a 100

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # --- Iconos (cacheados para evitar regenerarlos en cada toggle) ---
        self._icon_play: QIcon = qta.icon("fa5s.play", color=Colors.PRIMARY)
        self._icon_pause: QIcon = qta.icon("fa5s.pause", color=Colors.PRIMARY)
        self._icon_reset: QIcon = qta.icon("fa5s.undo-alt", color=Colors.PRIMARY)

        # --- Timer de animación -------------------------------------------
        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._on_timer_tick)

        # --- Construcción de la UI ----------------------------------------
        self._build_ui()

        # Estado inicial: pausado en 0%
        self._update_label(0)

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        """Construye el layout horizontal compacto con los controles."""
        self.setMaximumHeight(_WIDGET_MAX_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Botón Play/Pause (toggle)
        self._btn_play = QPushButton(self)
        self._btn_play.setIcon(self._icon_play)
        self._btn_play.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self._btn_play.setToolTip("Iniciar animación de aplicación de carga")
        self._btn_play.clicked.connect(self._toggle_play)
        layout.addWidget(self._btn_play)

        # Slider 0-100
        self._slider = QSlider(Qt.Horizontal, self)
        self._slider.setRange(0, 100)
        self._slider.setValue(0)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(25)
        self._slider.setToolTip(
            "Aplicación progresiva de la carga: "
            "0% = sin carga, 100% = carga completa"
        )
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        # Label "X %"
        self._label = QLabel("0 %", self)
        self._label.setMinimumWidth(48)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            f"color: {Colors.PRIMARY}; font-weight: 600;"
        )
        layout.addWidget(self._label)

        # Botón Reset
        self._btn_reset = QPushButton(self)
        self._btn_reset.setIcon(self._icon_reset)
        self._btn_reset.setFixedSize(_BUTTON_SIZE, _BUTTON_SIZE)
        self._btn_reset.setToolTip("Volver a configuración sin deformar (0%)")
        self._btn_reset.clicked.connect(self.reset)
        layout.addWidget(self._btn_reset)

    # ----------------------------------------------------------- API pública

    def set_percent(self, percent: float) -> None:
        """Setea el porcentaje manualmente sin disparar la señal.

        Útil para sincronizar el widget con un estado externo sin generar
        un re-render redundante.
        """
        value = int(round(max(0.0, min(100.0, float(percent)))))
        # Bloqueamos las señales del slider para no emitir progress_changed.
        was_blocked = self._slider.blockSignals(True)
        try:
            self._slider.setValue(value)
        finally:
            self._slider.blockSignals(was_blocked)
        self._update_label(value)

    def is_playing(self) -> bool:
        """Devuelve True si el timer de animación está activo."""
        return self._timer.isActive()

    def play(self) -> None:
        """Inicia la animación en bucle continuo (0 -> 100 -> 0 -> 100...)."""
        if self._timer.isActive():
            return
        self._timer.start()
        self._btn_play.setIcon(self._icon_pause)
        self._btn_play.setToolTip("Pausar animación")

    def pause(self) -> None:
        """Pausa la animación, manteniendo el porcentaje actual."""
        if not self._timer.isActive():
            return
        self._timer.stop()
        self._btn_play.setIcon(self._icon_play)
        self._btn_play.setToolTip("Iniciar animación de aplicación de carga")

    def reset(self) -> None:
        """Vuelve a 0%, deteniendo la animación si estaba en curso.

        Esta acción SÍ emite ``progress_changed(0)`` para que el plot vuelva
        al estado sin deformar.
        """
        self.pause()
        # Forzamos emisión incluso si ya estaba en 0: si el slider está en 0
        # setValue(0) no dispara valueChanged, así que emitimos manualmente.
        if self._slider.value() == 0:
            self._update_label(0)
            self.progress_changed.emit(0.0)
        else:
            self._slider.setValue(0)  # dispara _on_slider_changed -> emit

    # -------------------------------------------------------- Slots internos

    def _toggle_play(self) -> None:
        """Alterna entre play y pause según el estado actual."""
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def _on_timer_tick(self) -> None:
        """Avanza el slider en cada tick; al llegar a 100 vuelve a 0."""
        current = self._slider.value()
        next_value = current + _STEP_PER_TICK
        if next_value > 100:
            next_value = 0  # bucle continuo: 0 -> 100 -> 0 -> 100
        self._slider.setValue(next_value)

    def _on_slider_changed(self, value: int) -> None:
        """Maneja cualquier cambio del slider (manual o por timer)."""
        self._update_label(value)
        self.progress_changed.emit(float(value))

    def _update_label(self, value: int) -> None:
        """Actualiza el texto del label de porcentaje."""
        self._label.setText(f"{value} %")
