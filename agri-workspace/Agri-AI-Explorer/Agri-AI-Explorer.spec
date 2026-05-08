# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPEC).resolve().parent

torch_binaries = collect_dynamic_libs('torch')
customtkinter_datas = collect_data_files('customtkinter')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=customtkinter_datas + [
        ('assets', 'assets'),
        ('media/videos/cnn_anim/1080p60/CNNWheatDisease.mp4', 'media/videos/cnn_anim/1080p60'),
    ],
    hiddenimports=['torch', 'torch.nn'] + collect_submodules('unittest'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'manim', 'sphinx', 'sphinxcontrib',
        'torchvision', 'torchaudio',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'jupyter', 'jupyter_client', 'jupyter_core', 'notebook',
        'pytest', 'nose',
        'docutils', 'babel',
        'paramiko',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Agri-AI-Explorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Agri-AI-Explorer',
)
