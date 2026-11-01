# PackProof prototype
    pip install -r requirements.txt
    streamlit run app.py
Tabs: 1 design & run · 2 compare + find recommended · 3 safe-zone map · 4 bench calibration · 5 evidence & run record.
Swap the cell by uploading a JSON like `cell_hx32700.json`. Status: simulation is datasheet-based; nothing is bench-measured until you enter data in tab 4.
Limitation: the charge-temperature table and cycle-life anchors are still those of the HX cell inside model.py; a new cell needs them edited.
