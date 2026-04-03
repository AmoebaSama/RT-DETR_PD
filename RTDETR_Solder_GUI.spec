# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['launch_rtdetr_app.py'],
    pathex=[],
    binaries=[],
    datas=[('rtdetr\\web\\templates', 'rtdetr\\web\\templates'), ('rtdetr\\web\\static', 'rtdetr\\web\\static'), ('rtdetr\\runs\\solder_defects_rtdetr\\weights\\best.pt', 'rtdetr\\runs\\solder_defects_rtdetr\\weights'), ('rtdetr\\runs\\smoke_test\\weights\\best.pt', 'rtdetr\\runs\\smoke_test\\weights')],
    hiddenimports=[],
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
    name='RTDETR_Solder_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    upx=True,
    upx_exclude=[],
    name='RTDETR_Solder_GUI',
)
