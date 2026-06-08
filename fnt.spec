# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

hiddenimports = []
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('sam2')
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')
hiddenimports += collect_submodules('segmentation_models_pytorch')
hiddenimports += ['scipy.special.cython_special', 'sklearn.utils._typedefs']

datas = []
datas += [('icons', 'icons')]
datas += [('fnt/fed3/fed3_image.svg', 'fnt/fed3')]
datas += collect_data_files('ultralytics')
datas += collect_data_files('sam2')
datas += collect_data_files('segmentation_models_pytorch')
datas += copy_metadata('fnt')

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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='fnt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons/fnt_icon.ico'],
)
