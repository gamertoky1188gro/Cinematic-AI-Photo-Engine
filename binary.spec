# -*- mode: python ; coding: utf-8 -*-

import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
ROOT = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(ROOT, 'binary.py')],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (os.path.join(ROOT, 'engine'), 'engine'),
        (os.path.join(ROOT, 'models'), 'models'),
        (os.path.join(ROOT, 'Image-Adaptive-3DLUT'), 'Image-Adaptive-3DLUT'),
        (os.path.join(ROOT, 'frontend (binary)', 'dist'), 'frontend (binary)/dist'),
    ],
    hiddenimports=[
        'pywebview',
        'clr_loader',
        'pythonnet',
        'bottle',
        'proxy_tools',
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        'torchvision.transforms',
        'torchvision.transforms.functional',
        'cv2',
        'numpy',
        'scipy',
        'scipy.ndimage',
        'PIL',
        'PIL.Image',
        'rawpy',
        'engine',
        'engine.pipeline',
        'engine.analyzer',
        'engine.intelligence',
        'engine.hybrid_intelligence',
        'engine.render_plan',
        'engine.evaluate',
        'engine.stages',
        'engine.stages.adaptive_lut',
        'engine.stages.film_transform',
        'engine.stages.optical_renderer',
        'engine.stages.geometry_renderer',
        'models.parameter_predictor',
        'models_modern',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CinematicAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CinematicAI',
)
