import pyreadr
from pyreadr._pyreadr_parser import PyreadrParser
import os
import gc

parser = PyreadrParser()
rdata_path = os.path.abspath("TEP_Faulty_Testing.RData")
parser.parse(os.fsencode(rdata_path))

print(f"Number of tables: {len(parser.table_data)}")
table = parser.table_data[0]
print(f"Table name: {table.name}")
print(f"Table attributes: {[attr for attr in dir(table) if not attr.startswith('_')]}")

if hasattr(table, "columns"):
    print(f"Number of columns: {len(table.columns)}")
    for col in table.columns[:5]:
        print(f"Col name: {col.name}, type: {type(col.data)}, len: {len(col.data) if hasattr(col.data, '__len__') else 'N/A'}")
