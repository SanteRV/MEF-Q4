================================================================
  MEF Q4 — Aplicativo de Análisis por Elementos Finitos
  Trabajo de Tesis · Ingeniería Civil
  Versión 2.1
================================================================

QUÉ ES
------
Aplicativo didáctico para análisis por elementos finitos con el
elemento cuadrilátero bilineal Q4 (4 nodos, 2 GDL por nodo,
isoparamétrico). Resuelve problemas planos (tensión/deformación
plana) y muestra paso a paso las 15 etapas del cálculo: funciones
de forma, puntos de Gauss, matriz D, jacobiano, matriz B,
ensamblaje, condiciones de borde, desplazamientos, reacciones,
esfuerzos en puntos de Gauss y en nodos.


NOVEDADES DE ESTA VERSIÓN
-------------------------
- NUEVA pestaña "Vista 3D": el modelo Q4 plano se puede ver con
  cámara 3D orbital (rotar, hacer pan, zoom, ver desde cualquier
  ángulo). El elemento sigue siendo el mismo Q4 plano, lo que
  cambia es la presentación: placa en perspectiva, deformada
  superpuesta en rojo, apoyos como cubos verdes, cargas como
  flechas naranja, mapa de esfuerzos coloreado por elemento.
- Funciones de forma N1..N4 ahora se muestran como superficies
  3D (cada Ni vale 1 en su propio nodo y 0 en los otros tres,
  evidenciando la partición de la unidad).
- Diagrama de puntos de Gauss en perspectiva 3D con líneas de
  cuadratura y nodos de referencia.
- Correcciones del documento aplicadas:
    * PDF con tablas redimensionadas (no se sale del margen).
    * Tablas con scroll interno cuando hay muchos nodos.
    * Materiales mostrados en notación ingenieril (GPa/MPa).
    * Excel con dos formatos: por pasos o por categorías.


INSTRUCCIONES DE USO
--------------------
1. Descomprime este ZIP en cualquier carpeta (por ejemplo el
   Escritorio).
2. Entra a la carpeta "MEF_Q4" y haz DOBLE CLIC en:
        MEF_Q4.exe
3. La primera vez tarda 8-15 segundos en abrir.
4. No requiere instalación, Python ni librerías externas.


REQUISITOS
----------
- Windows 10 o superior, 64 bits.
- Aproximadamente 500 MB libres (incluye OpenGL para la vista 3D).
- Tarjeta gráfica con soporte OpenGL 2.0 o superior (cualquiera
  de los últimos 15 años sirve).
- NO requiere conexión a internet.


PESTAÑAS DEL APLICATIVO
-----------------------
1. "Paso a paso": navegación por las 15 etapas didácticas del
   cálculo. Para cada paso se muestra la fórmula, la sustitución
   numérica, el resultado en tablas y matrices, y los gráficos
   correspondientes (funciones de forma como superficies 3D,
   puntos de Gauss en perspectiva, malla, deformada, mapa de
   esfuerzos, esfuerzos principales).
2. "Editor gráfico": lienzo plano estilo SAP2000 para crear
   modelos con el mouse. Modos: seleccionar, añadir nodo, añadir
   Q4, modo rectángulo (clic + arrastre para crear 4 nodos y un
   Q4), apoyos, cargas, eliminar. Atajos de teclado: S/N/Q/R/A/L/D.
   Arrastrar para mover, deshacer/rehacer (Ctrl+Z / Ctrl+Y),
   selección por rectángulo (Ctrl + arrastre).
3. "Vista 3D": el modelo Q4 plano visto con cámara 3D orbital.
   Rotar (botón izquierdo + arrastre), desplazar (botón derecho
   + arrastre), zoom (rueda del mouse). Toggles para mostrar
   u ocultar malla original, deformada, caras semitransparentes,
   apoyos, cargas y etiquetas de nodos. Slider de escala para la
   deformada. Selector de campo para colorear según σx, σy, τxy,
   σ_VM, σ1 o σ2. Botón "Centrar vista".


HERRAMIENTAS Y EXPORTACIONES
----------------------------
Menú "Herramientas":
  - Estudio de convergencia automático.
  - Comparación con solución analítica.
  - Círculo de Mohr (estado tensional en un punto).

Menú "Archivo":
  - Nuevo / Abrir / Guardar proyecto (.json).
  - Exportar reporte de análisis (PDF).
  - Exportar a Excel (por pasos o por categorías).

Manual teórico (PDF estático) disponible desde el menú Ayuda.


CONTROLES ÚTILES
----------------
- Ctrl+N           Nuevo proyecto.
- Ctrl+O           Abrir proyecto.
- Ctrl+S           Guardar proyecto.
- Ctrl+P           Exportar reporte de análisis (PDF).
- Ctrl+E           Exportar a Excel.
- Ctrl+K           Estudio de convergencia.
- Ctrl+A           Comparar con solución analítica.
- Ctrl+M           Círculo de Mohr.
- Ctrl+Z / Ctrl+Y  Deshacer / Rehacer (en editor gráfico).
- Ctrl+Izq/Der     Paso anterior / siguiente.


PROBLEMAS COMUNES
-----------------
- Windows SmartScreen lo bloquea: haz clic en "Más información"
  y luego en "Ejecutar de todos modos". Es seguro; es un
  ejecutable no firmado digitalmente (no se paga el certificado
  para una tesis).
- El antivirus lo marca como sospechoso: igual que el caso
  anterior, los ejecutables generados por PyInstaller son
  a veces falsos positivos. Si fuera necesario, excluir la
  carpeta MEF_Q4 del antivirus.
- Tarda mucho en abrir: la primera vez Windows carga todos los
  DLL. A partir de la segunda apertura es casi instantáneo.
- La vista 3D se ve en negro: la tarjeta gráfica es muy antigua
  y no tiene soporte OpenGL 2.0 o superior. Probar a actualizar
  el controlador (driver).


CONTACTO
--------
Autor: Manuel Carbajal

================================================================
