"""MuseStudio — connect, stream, record and visualize Muse S Athena EEG/fNIRS data.

V1 scope: Direct BLE via OpenMuse (producer) -> LSL -> in-app reader (consumer)
-> live plot + CSV recording. See fnt/musestudio/musestudio_pyqt.py for the GUI.

Note: OpenMuse's decoding (especially fNIRS) is reverse-engineered and
experimental; it is not affiliated with or endorsed by InteraXon.
"""
