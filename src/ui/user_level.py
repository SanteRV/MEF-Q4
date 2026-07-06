"""Sistema de nivel de usuario.

Define dos perfiles que controlan qué tan didáctica/detallada es la UI:

- NOVATO         → descripciones largas, fórmulas teóricas explicadas,
                   ayuda contextual visible, tooltips siempre activos
- EXPERIMENTADO  → descripciones cortas, solo matrices y resultados,
                   UI compacta y rápida
"""
from __future__ import annotations
from enum import Enum
import json
from pathlib import Path


class UserLevel(str, Enum):
    NOVATO = "novato"
    EXPERIMENTADO = "experimentado"

    @property
    def label(self) -> str:
        return {
            UserLevel.NOVATO: "Novato",
            UserLevel.EXPERIMENTADO: "Experimentado",
        }[self]

    @property
    def description(self) -> str:
        return {
            UserLevel.NOVATO:
                "Modo didáctico: descripciones detalladas, fórmulas explicadas, "
                "tooltips visibles, ayuda contextual en cada paso.",
            UserLevel.EXPERIMENTADO:
                "Modo compacto: solo matrices y resultados, descripciones cortas, "
                "UI densa para trabajo rápido.",
        }[self]


# Singleton global del nivel actual
_current: UserLevel = UserLevel.NOVATO
_PREF_PATH = Path.home() / ".trabajo_tesis_user_level.json"


def get_level() -> UserLevel:
    """Devuelve el nivel actual."""
    return _current


def set_level(level: UserLevel) -> None:
    """Setea y persiste el nivel."""
    global _current
    _current = level
    try:
        with open(_PREF_PATH, "w", encoding="utf-8") as f:
            json.dump({"level": level.value}, f)
    except Exception:
        pass


def load_saved_level() -> UserLevel:
    """Recupera el nivel guardado en preferencias (o NOVATO por defecto)."""
    global _current
    try:
        if _PREF_PATH.exists():
            with open(_PREF_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                _current = UserLevel(data.get("level", UserLevel.NOVATO.value))
    except Exception:
        _current = UserLevel.NOVATO
    return _current


def is_novato() -> bool:
    return _current == UserLevel.NOVATO


def is_experimentado() -> bool:
    return _current == UserLevel.EXPERIMENTADO
