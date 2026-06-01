# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = []
# Deep learning libraries often have hidden imports
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('sam2')
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')
hiddenimports += collect_submodules('segmentation_models_pytorch')
hiddenimports += ['scipy.special.cython_special', 'sklearn.utils._typedefs']

datas = []
# Include our icons folder
datas += [('icons', 'icons')]
# Include data files for deep learning libraries (configs, etc.)
datas += collect_data_files('ultralytics')
datas += collect_data_files('sam2')
datas += collect_data_files('segmentation_models_pytorch')

a = Analysis(
    ['fnt/gui_pyqt.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fnt-gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to True so users can see error logs if it crashes
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/fnt_icon.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fnt-gui',
)
