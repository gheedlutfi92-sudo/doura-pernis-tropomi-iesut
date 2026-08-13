"""
add_ehsa_fields.py
==================
Adds obs_date (Date) and pixel_id (Text) fields to all 4 IESUT GDB
feature classes so they can be used as input to the Space Time Cube
tool for EHSA (Emerging Hot Spot Analysis).

Run this in PyCharm (with ArcGIS Pro Python environment selected).

Fields added:
  obs_date  - Date field: first day of each year/month (e.g. 2018-05-01)
  pixel_id  - Text field: "lon_lat" string identifying each unique pixel
               (e.g. "44.4012_33.2567") — used as Location ID in Space Time Cube
"""

import arcpy
import datetime

GDB = r"C:\Users\HP\Desktop\proposal& dissertation\dissertation_final\dissertation_final\dissertation_final.gdb"

FEATURE_CLASSES = [
    "IESUT_doura_no2",
    "IESUT_doura_so2",
    "IESUT_pernis_no2",
    "IESUT_pernis_so2",
]


def existing_field_names(fc_path):
    return [f.name.lower() for f in arcpy.ListFields(fc_path)]


def add_field_if_missing(fc_path, field_name, field_type, length=None):
    if field_name.lower() not in existing_field_names(fc_path):
        if length:
            arcpy.management.AddField(fc_path, field_name, field_type,
                                      field_length=length)
        else:
            arcpy.management.AddField(fc_path, field_name, field_type)
        print(f"    Added field: {field_name}")
    else:
        print(f"    Field already exists (will recalculate): {field_name}")


for fc_name in FEATURE_CLASSES:
    fc_path = f"{GDB}\\{fc_name}"
    print(f"\n{'='*50}")
    print(f"Processing: {fc_name}")
    print(f"{'='*50}")

    # --- obs_date ---
    add_field_if_missing(fc_path, "obs_date", "DATE")
    count = 0
    with arcpy.da.UpdateCursor(fc_path, ["year", "month", "obs_date"]) as cur:
        for row in cur:
            row[2] = datetime.date(int(row[0]), int(row[1]), 1)
            cur.updateRow(row)
            count += 1
    print(f"    obs_date calculated for {count:,} rows")

    # --- pixel_id ---
    add_field_if_missing(fc_path, "pixel_id", "TEXT", length=30)
    count = 0
    with arcpy.da.UpdateCursor(fc_path, ["SHAPE@XY", "pixel_id"]) as cur:
        for row in cur:
            xy = row[0]
            row[1] = f"{xy[0]:.4f}_{xy[1]:.4f}"
            cur.updateRow(row)
            count += 1
    print(f"    pixel_id calculated for {count:,} rows")

print(f"\n{'='*50}")
print("All 4 feature classes updated successfully.")
print("You can now run Space Time Cube from Defined Locations.")
print(f"{'='*50}")
