import gc
import pyreadr
from pyreadr._pyreadr_parser import PyreadrParser

print("Parsing RData with PyreadrParser...")
parser = PyreadrParser()
parser.parse_file("TEP_Faulty_Testing.RData")

print(f"Number of objects found: {len(parser.objects)}")
for obj in parser.objects:
    print(f"Object name: {obj.name}, type: {obj.type}, rows: {obj.row_count}, cols: {obj.col_count}")
