import os
import gc
import numpy as np
import pandas as pd
from pyreadr._pyreadr_parser import PyreadrParser

TEP_COLS = ['faultNumber', 'simulationRun', 'sample'] + [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{j}' for j in range(1, 12)]

def extract_faults_memory_efficient():
    tep_dir = os.path.dirname(os.path.abspath(__file__))
    rdata_path = os.path.join(tep_dir, "TEP_Faulty_Testing.RData")

    print(f"Parsing RData file: {rdata_path} ...")
    parser = PyreadrParser()
    parser.parse(os.fsencode(rdata_path))

    if not parser.table_data:
        raise RuntimeError("No tables found in RData file!")

    table = parser.table_data[0]
    col_arrays = list(table.columns)
    print(f"Found table '{table.name}' with {len(col_arrays)} columns.")

    if len(col_arrays) == 55:
        col_names = TEP_COLS
    else:
        col_names = [str(c) for c in table.column_names]

    data_dict = {col_names[i]: col_arrays[i] for i in range(len(col_names))}

    fault_col = 'faultNumber' if 'faultNumber' in data_dict else col_names[0]
    fault_vals = np.asarray(data_dict[fault_col])
    unique_faults = np.unique(fault_vals)
    print(f"Using fault column: '{fault_col}' with unique fault IDs: {unique_faults}")

    for fault_id in unique_faults:
        mask = (fault_vals == fault_id)
        sub_dict = {}
        for c in col_names:
            sub_dict[c] = np.asarray(data_dict[c])[mask]

        sub_df = pd.DataFrame(sub_dict)
        out_name = f"TEP_Faulty_Testing_fault{int(fault_id)}.pkl"
        out_path = os.path.join(tep_dir, out_name)
        sub_df.to_pickle(out_path)
        print(f"Saved {out_name} (shape: {sub_df.shape})")
        del sub_dict, sub_df
        gc.collect()

    print("\nAll 20 fault files extracted and column-named successfully!")

if __name__ == "__main__":
    extract_faults_memory_efficient()
