from openvsp import openvsp as vsp

vsp.ClearVSPModel()
vsp.ReadVSPFile(r"C:\VSP\Development\PSO_PYTHON_WING\cessna210.vsp3")

geom_ids = vsp.FindGeoms()

print("\n[debug] Geometrias e seus Sets:")
for gid in geom_ids:
    sets = vsp.GetGeomSet(gid)
    print(f"{gid} → {sets}")
