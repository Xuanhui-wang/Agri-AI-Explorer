# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPEC).resolve().parent

torch_binaries = collect_dynamic_libs('torch')
customtkinter_datas = collect_data_files('customtkinter')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=torch_binaries,
    datas=customtkinter_datas + [
        ('assets', 'assets'),
        (str((ROOT.parent / 'media' / 'videos' / 'cnn_anim' / '1080p60' / 'partial_movie_files' / 'CNNWheatDisease' / 'uncached_00000.mp4')), 'media/videos/cnn_anim/1080p60/partial_movie_files/CNNWheatDisease'),
    ],
    hiddenimports=['torch', 'torch.nn'],
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
    a.datas,
    [],
    exclude_binaries=False,
    name='Agri-AI-Explorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
