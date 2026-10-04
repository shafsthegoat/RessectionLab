"""Arithmetic/XML controls only; never import Gmsh or launch a mesher/solver."""
import copy
import gzip
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("mechanics_hbe_mesh", ROOT/"scripts/mechanics_hbe_mesh.py")
mesh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mesh)


@pytest.fixture
def protocol():
    return mesh.read_protocol(ROOT/"manifests/experiments/hbe-01-03-mechanics-poc-v1.json")


def arithmetic_hex(protocol):
    """One inscribed rectangular hex, deliberately NOT a specimen mesh."""
    R, H = (protocol["geometry"][key] for key in ("radius_m", "height_m"))
    X = mesh.HEX_SIGNS*np.array([R/math.sqrt(2), R/math.sqrt(2), H/2])
    X[:, 2] += H/2
    cells = np.arange(1,9)[None,:]
    faces = cells[0, mesh.HEX_FACES]
    boundaries = {}
    for name, rows in (("bottom",faces[:1]),("top",faces[1:2]),("side",faces[2:])):
        boundaries[name] = {"node_ids":sorted(set(rows.flat)), "faces_quad4":rows.tolist(), "gmsh_surface_tags":[]}
    return {"schema_version":mesh.VERSION,"indexing":"one_based","mesh_N":4,
        "geometry":protocol["geometry"],"rest_nodes_m":X.tolist(),"node_ids":list(range(1,9)),
        "gmsh_node_ids":list(range(101,109)),"element_ids":[1],"gmsh_element_ids":[99],
        "elements_hex8":cells.tolist(),"boundaries":boundaries}


def test_import_never_imports_optional_gmsh():
    assert "gmsh" not in sys.modules


def test_protocol_hash_is_exact_and_mutation_stops(tmp_path, protocol):
    path = tmp_path/"protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="protocol bytes"):
        mesh.read_protocol(path)


def test_ordering_is_derived_from_reference_vertices_and_not_assumed():
    permutation = [5,2,7,0,6,3,1,4]
    local = mesh.HEX_SIGNS[permutation]
    recovered = mesh.hex8_permutation(local)
    np.testing.assert_array_equal(local[recovered], mesh.HEX_SIGNS)
    with pytest.raises(ValueError): mesh.hex8_permutation(local/2)
    local[0] = local[1]
    with pytest.raises(ValueError): mesh.hex8_permutation(local)


def test_sparse_gmsh_ids_map_to_explicit_febio_ids_without_orphan_cad_nodes(protocol):
    fixture=arithmetic_hex(protocol)
    original_ids=np.array([103,107,113,127,131,137,139,149])
    permutation=np.array([5,2,7,0,6,3,1,4])
    all_ids=np.append(original_ids[::-1],997)
    coordinates=np.vstack((np.array(fixture["rest_nodes_m"])[::-1],[0,0,0]))
    surface_ids={"bottom":[11],"top":[12],"side":[13]}
    surface_rows={surface_ids[name][0]:original_ids[np.array(row["faces_quad4"])-1].reshape(-1)
                  for name,row in fixture["boundaries"].items()}
    def get_elements(dimension,tag=-1):
        if dimension==3:return [5],[np.array([71])],[original_ids[permutation]]
        rows=surface_rows[tag]
        return [3],[np.arange(len(rows)//4)+1],[rows]
    api=SimpleNamespace(getElements=get_elements,
        getElementProperties=lambda kind:("Hexahedron 8",3,1,8,mesh.HEX_SIGNS[permutation].reshape(-1),8),
        getNodes=lambda:(all_ids,coordinates.reshape(-1),[]))
    extracted=mesh.extract_mesh(SimpleNamespace(model=SimpleNamespace(mesh=api)),4,protocol,
                                {"surface_tags":surface_ids})
    assert extracted["gmsh_node_ids"]==original_ids.tolist()
    assert extracted["node_ids"]==list(range(1,9))
    assert extracted["elements_hex8"]==[list(range(1,9))]
    assert extracted["gmsh_element_ids"]==[71]
    assert extracted["unused_geometric_node_ids"]==[997]
    np.testing.assert_array_equal(extracted["rest_nodes_m"],fixture["rest_nodes_m"])
    assert mesh.boundary_topology(np.array(extracted["elements_hex8"]),extracted["boundaries"])["exterior_face_count"]==6


def test_hex_volume_scaled_jacobian_and_positive_affine_map(protocol):
    fixture = arithmetic_hex(protocol)
    X = np.asarray(fixture["rest_nodes_m"])
    cells = np.asarray(fixture["elements_hex8"])
    determinants, scaled = mesh.rest_jacobians(X,cells)
    R,H = protocol["geometry"]["radius_m"],protocol["geometry"]["height_m"]
    assert determinants.shape == scaled.shape == (1,17)
    np.testing.assert_allclose(determinants, R*R*H/4, rtol=2e-15)
    np.testing.assert_allclose(scaled,1,rtol=2e-15)
    assert determinants[:,:8].sum() == pytest.approx(2*R*R*H,rel=2e-15)
    transform = np.array([[1.1,.2,0],[0,.9,0],[0,0,1.03]])
    moved,_ = mesh.rest_jacobians(X@transform.T+[.2,.3,-.1],cells)
    np.testing.assert_allclose(moved/determinants,np.linalg.det(transform),rtol=1e-12)


def test_inversion_and_corner_warp_cannot_hide_in_average_volume(protocol):
    fixture = arithmetic_hex(protocol)
    X,cells=np.array(fixture["rest_nodes_m"]),np.array(fixture["elements_hex8"])
    inverted,_=mesh.rest_jacobians(X,cells[:,[1,0,3,2,5,4,7,6]])
    assert np.all(inverted < 0)
    X[6]=X[0]
    determinants,_=mesh.rest_jacobians(X,cells)
    assert determinants.min() < 0
    assert determinants[:,:8].sum() > 0


@pytest.mark.parametrize("fault", ["float_connectivity","missing_vertex","repeated_vertex","nan"])
def test_malformed_connectivity_or_positions_fail(protocol,fault):
    fixture=arithmetic_hex(protocol)
    X,cells=np.array(fixture["rest_nodes_m"]),np.array(fixture["elements_hex8"])
    if fault=="float_connectivity":cells=cells.astype(float)
    elif fault=="missing_vertex":cells[0,0]=9
    elif fault=="repeated_vertex":cells[0,0]=cells[0,1]
    else:X[0,0]=np.nan
    with pytest.raises(ValueError):mesh.rest_jacobians(X,cells)


def test_actual_exterior_tags_and_outward_orientation(protocol):
    fixture=arithmetic_hex(protocol)
    result=mesh.boundary_topology(fixture["elements_hex8"],fixture["boundaries"])
    assert result["exterior_face_count"]==6 and result["interior_face_count"]==0
    assert result["connected_cell_count"]==1
    X=np.array(fixture["rest_nodes_m"]);center=X.mean(axis=0)
    for faces in result["outward_faces"].values():
        for face in faces:
            points=X[np.array(face)-1]
            assert np.cross(points[1]-points[0],points[2]-points[1])@(points.mean(axis=0)-center)>0


@pytest.mark.parametrize("fault",["missing_face","duplicate_tag","wrong_nodes","interior_crack"])
def test_missing_or_conflicting_boundary_evidence_is_rejected(protocol,fault):
    fixture=arithmetic_hex(protocol)
    if fault=="missing_face":fixture["boundaries"]["side"]["faces_quad4"].pop()
    elif fault=="duplicate_tag":fixture["boundaries"]["side"]["faces_quad4"].append(fixture["boundaries"]["top"]["faces_quad4"][0])
    elif fault=="wrong_nodes":fixture["boundaries"]["top"]["node_ids"].pop()
    else:fixture["elements_hex8"].append(list(range(9,17)))
    with pytest.raises(ValueError):mesh.boundary_topology(fixture["elements_hex8"],fixture["boundaries"])


def test_actual_sag_and_polygon_volume_do_not_claim_ideal_cad_geometry(protocol):
    fixture=arithmetic_hex(protocol)
    measured=mesh.mesh_quality(fixture,protocol)
    assert not measured["passed"] # One arithmetic hex is not the declared96-cell mesh.
    assert measured["checks"]["declared_cell_count"] is False
    assert measured["relative_volume_error"]==pytest.approx(1-2/math.pi)
    assert measured["maximum_radial_boundary_sag_over_R"]==pytest.approx(1-1/math.sqrt(2))
    fixture["mesh_N"]=12
    fine=mesh.mesh_quality(fixture,protocol)
    assert fine["checks"]["finest_volume"] is False
    assert fine["checks"]["finest_boundary_sag"] is False


def test_boundary_tag_cannot_be_silently_inferred_from_wrong_geometry(protocol):
    fixture=arithmetic_hex(protocol)
    fixture["boundaries"]["top"],fixture["boundaries"]["bottom"]=fixture["boundaries"]["bottom"],fixture["boundaries"]["top"]
    with pytest.raises(ValueError,match="boundary tags"):
        mesh.mesh_quality(fixture,protocol)


@pytest.mark.parametrize("branch",["torsion_neg","torsion_pos"])
@pytest.mark.parametrize("steps",[60,120])
def test_exact_rotation_at_every_step_preserves_radius_and_is_not_final_chord(protocol,branch,steps):
    fixture=arithmetic_hex(protocol)
    loading=mesh.prescribed_motion(fixture,branch,steps,protocol)
    initial=np.asarray(fixture["rest_nodes_m"])[loading["top_node_ids"]-1]
    current=initial[None,:,:]+loading["top_displacement_m"]
    np.testing.assert_allclose(np.linalg.norm(current[:,:,:2],axis=2),protocol["geometry"]["radius_m"],atol=2e-18,rtol=0)
    np.testing.assert_array_equal(loading["top_displacement_m"][:,:,2],0)
    halfway=steps//2
    assert np.max(np.abs(loading["top_displacement_m"][halfway]-.5*loading["top_displacement_m"][-1]))>1e-5
    assert loading["load_coordinate_units"]=="rad"
    assert np.sign(loading["load_coordinate"][-1])==(-1 if branch.endswith("neg") else 1)


@pytest.mark.parametrize("branch,sign",[("compression",-1),("tension",1)])
def test_axial_loading_leaves_top_in_plane_motion_zero(protocol,branch,sign):
    loading=mesh.prescribed_motion(arithmetic_hex(protocol),branch,60,protocol)
    np.testing.assert_array_equal(loading["top_displacement_m"][:,:,:2],0)
    np.testing.assert_allclose(loading["top_displacement_m"][-1,:,2],sign*.15*protocol["geometry"]["height_m"])


def test_deck_matches_material_formulation_fixed_steps_and_reaction_contract(protocol):
    fixture=arithmetic_hex(protocol)
    xml,record=mesh.specimen_deck(fixture,"torsion_pos",60,1000.,protocol)
    root=ET.fromstring(xml)
    assert root.attrib=={"version":"4.0"}
    assert root.findtext("Control/plot_level")=="PLOT_NEVER"
    assert root.findtext("Control/output_level")=="OUTPUT_MAJOR_ITRS"
    assert root.findtext("Control/time_stepper/max_retries")=="0"
    assert float(root.findtext("Control/time_stepper/dtmin"))==float(root.findtext("Control/time_stepper/dtmax"))==1/60
    assert root.find("Control/solver/linear_solver").get("type")=="skyline"
    assert root.findtext("Control/solver/rtol")=="1e-08"
    assert float(root.findtext("Control/solver/min_residual"))==(1e-10*1000*.004**2)**2
    assert root.findtext("Material/material/c1")=="2000"
    assert float(root.findtext("Material/material/k"))==149000/3
    assert root.findtext("Material/material/pressure_model")=="1"
    assert root.find("MeshDomains/SolidDomain").get("type")=="three-field-solid"
    assert root.findtext("MeshDomains/SolidDomain/laugon")=="0"
    assert len(root.findall("Boundary/bc"))==15
    assert all(bc.get("type")=="prescribed displacement" for bc in root.findall("Boundary/bc"))
    assert all(bc.findtext("relative")=="0" for bc in root.findall("Boundary/bc"))
    assert {s.get("name") for s in root.findall("Mesh/Surface")}=={"top","bottom","side"}
    assert not any(root.findall(tag) for tag in ("Loads","Contact","Rigid","Initial"))
    assert root.find("Output/logfile/node_data").get("data")==mesh.NODE_FIELDS
    assert root.find("Output/logfile/element_data").get("data")==mesh.ELEMENT_FIELDS
    assert root.find("Output/logfile/node_data").get("format") is None
    assert root.find("Output/plotfile") is None
    assert len(root.findall("LoadData/load_controller"))==record["load_controller_count"]
    for curve in root.findall("LoadData/load_controller"):
        assert len(curve.findall("points/pt"))==61
        assert curve.findtext("interpolate")=="LINEAR"
    assert record["times"]==np.linspace(0,1,61).tolist()


def test_deck_exact_rotation_controllers_and_modulus_scaling(protocol):
    fixture=arithmetic_hex(protocol)
    base,base_record=mesh.specimen_deck(fixture,"torsion_pos",120,1000.,protocol)
    doubled,double_record=mesh.specimen_deck(fixture,"torsion_pos",120,2000.,protocol)
    b,d=ET.fromstring(base),ET.fromstring(doubled)
    assert ET.tostring(b.find("Boundary"))==ET.tostring(d.find("Boundary"))
    assert ET.tostring(b.find("LoadData"))==ET.tostring(d.find("LoadData"))
    assert float(d.findtext("Material/material/c1"))==2*float(b.findtext("Material/material/c1"))
    assert double_record["min_residual_N2"]==4*base_record["min_residual_N2"]
    prescribed=mesh.prescribed_motion(fixture,"torsion_pos",120,protocol)
    controllers={c.get("id"):c for c in b.findall("LoadData/load_controller")}
    for bc in b.findall("Boundary/bc"):
        name=bc.get("node_set")
        values=np.array([[float(v) for v in pt.text.split(",")] for pt in controllers[bc.find("value").get("lc")].findall("points/pt")])
        if name=="bottom":np.testing.assert_array_equal(values[:,1],0)
        else:
            node=int(name.split("_")[-1]);column=prescribed["top_node_ids"].tolist().index(node)
            axis="xyz".index(bc.findtext("dof"))
            np.testing.assert_array_equal(values[:,1],prescribed["top_displacement_m"][:,column,axis])


@pytest.mark.parametrize("branch",["compression","tension","torsion_neg","torsion_pos"])
@pytest.mark.parametrize("steps",[60,120])
def test_deck_bytes_and_loading_hash_survive_mesh_json_roundtrip(protocol,branch,steps):
    fixture=arithmetic_hex(protocol)
    expected_xml,expected_loading=mesh.specimen_deck(fixture,branch,steps,1375.,protocol)
    serialized=json.dumps(fixture,sort_keys=True,default=int)
    reloaded=json.loads(serialized)
    assert list(reloaded["boundaries"])==["bottom","side","top"]
    for order in itertools.permutations(("bottom","top","side")):
        reordered=copy.deepcopy(reloaded)
        reordered["boundaries"]={name:reloaded["boundaries"][name] for name in order}
        actual_xml,actual_loading=mesh.specimen_deck(reordered,branch,steps,1375.,protocol)
        assert actual_xml==expected_xml
        assert actual_loading==expected_loading
    assert json.dumps(fixture,sort_keys=True,default=int)==serialized


def test_closed_log_compression_is_lossless_bounded_and_retains_raw_on_failure(tmp_path):
    path=tmp_path/"nodes.log";data=b"finite signed raw reactions\n"*100
    path.write_bytes(data)
    receipt=mesh.compress_closed_log(path,maximum_raw_bytes=len(data),maximum_retained_bytes=4096)
    assert not path.exists() and receipt["decompressed_hash_verified"]
    assert gzip.decompress((tmp_path/receipt["archive_name"]).read_bytes())==data
    failure=tmp_path/"failed.log";failure.write_bytes(data)
    with pytest.raises(ValueError):mesh.compress_closed_log(failure,maximum_raw_bytes=1,maximum_retained_bytes=4096)
    assert failure.read_bytes()==data
    with pytest.raises(ValueError):mesh.compress_closed_log(failure,maximum_raw_bytes=len(data),maximum_retained_bytes=1)
    assert failure.read_bytes()==data


def test_closed_log_refuses_symlinks_and_existing_archive(tmp_path):
    path=tmp_path/"data.log";path.write_text("evidence")
    link=tmp_path/"alias.log";link.symlink_to(path)
    with pytest.raises(ValueError):mesh.compress_closed_log(link,maximum_raw_bytes=100,maximum_retained_bytes=100)
    path.with_name(path.name+".gz").write_bytes(b"earlier attempt")
    with pytest.raises(FileExistsError):mesh.compress_closed_log(path,maximum_raw_bytes=100,maximum_retained_bytes=100)
    assert path.read_text()=="evidence"


def fake_runtime(tmp_path,protocol):
    prefix=tmp_path/"private";prefix.mkdir()
    value={"status":"verified_ready_for_separate_meshing_release","version":"4.15.2",
        "wheel":{"sha256":protocol["mesher"]["sha256"]},
        "source_declaration":{"manifest_sha256":mesh.PROTOCOL_SHA256},
        "probe":{"child":{"status":"passed"}},"private_prefix":str(prefix),
        "interpreter":{"resolved_path":str(Path(sys.executable).resolve()),"sha256":mesh.sha256(Path(sys.executable).resolve())}}
    for name in ("module","library","license"):
        path=prefix/name;path.write_text("arithmetic-only identity fixture "+name)
        value[name]={"path":str(path),"sha256":mesh.sha256(path)}
    path=tmp_path/"runtime-receipt.json";path.write_text(json.dumps(value))
    return path,value


def test_runtime_binding_checks_module_library_interpreter_before_import(tmp_path,protocol):
    path,value=fake_runtime(tmp_path,protocol)
    assert mesh.verify_runtime(path,mesh.sha256(path),protocol)==value
    assert "gmsh" not in sys.modules
    Path(value["library"]["path"]).write_text("changed")
    with pytest.raises(ValueError,match="changed: library"):
        mesh.verify_runtime(path,mesh.sha256(path),protocol)
    with pytest.raises(ValueError,match="frozen private-runtime receipt"):
        mesh.verify_runtime(path,"0"*64,protocol)


def test_runtime_binding_rejects_prefix_escape_and_wrong_probe(tmp_path,protocol):
    path,value=fake_runtime(tmp_path,protocol)
    outside=tmp_path/"outside";outside.write_text("not private")
    value["module"]={"path":str(outside),"sha256":mesh.sha256(outside)}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="escaped"):
        mesh.verify_runtime(path,mesh.sha256(path),protocol)
    value["probe"]["child"]["status"]="failed";path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="successful probe"):
        mesh.verify_runtime(path,mesh.sha256(path),protocol)


def test_invalid_runtime_preserves_worker_failure_without_native_import(tmp_path,protocol):
    path,_=fake_runtime(tmp_path,protocol)
    output=tmp_path/"attempt"
    with pytest.raises(ValueError):mesh.prepare_level(4,protocol,path,"0"*64,output)
    receipt=json.loads((output/"receipt.json").read_text())
    assert receipt["status"]=="failed"
    assert receipt["gmsh_generation_calls"]==receipt["solver_calls"]==0
    assert receipt["curve_values_opened"] is False and "gmsh" not in sys.modules


def test_default_cli_is_only_validation_and_requires_source_identity(tmp_path,protocol,monkeypatch,capsys):
    path,_=fake_runtime(tmp_path,protocol)
    arguments=["mesh","--protocol",str(ROOT/"manifests/experiments/hbe-01-03-mechanics-poc-v1.json"),
        "--runtime-receipt",str(path),"--runtime-receipt-sha256",mesh.sha256(path),
        "--expected-source-sha256",mesh.sha256(mesh.__file__)]
    monkeypatch.setattr(sys,"argv",arguments)
    mesh.main()
    assert "no Gmsh import" in capsys.readouterr().out
    assert "gmsh" not in sys.modules
    arguments[-1]="0"*64
    with pytest.raises(ValueError,match="source differs"):mesh.main()


@pytest.mark.parametrize("fault",["wall","rss","active_output","total_output","observer"])
def test_mesh_supervisor_kills_and_reaps_on_every_declared_cap(tmp_path,protocol,monkeypatch,fault):
    raw=tmp_path/"raw";raw.mkdir()
    output=raw/"attempt"
    p=copy.deepcopy(protocol)
    p["budgets"]["sampled_process_family_rss_bytes"]=100
    p["budgets"]["each_active_run_output_bytes"]=20000
    p["budgets"]["generated_output_bytes"]=40000
    if fault=="active_output":p["budgets"]["each_active_run_output_bytes"]=1
    if fault=="total_output":p["budgets"]["generated_output_bytes"]=1
    class Process:
        pid=123456
        reaped=False
        def poll(self):return None
        def wait(self,timeout):self.reaped=True;return -9
    process=Process();kills=[]
    monkeypatch.setattr(mesh.subprocess,"Popen",lambda *a,**k:process)
    monkeypatch.setattr(mesh.os,"killpg",lambda pid,signal:kills.append((pid,signal)))
    ticks=iter(np.arange(0,100,.2))
    monkeypatch.setattr(mesh.time,"monotonic",lambda:float(next(ticks)))
    monkeypatch.setattr(mesh.time,"sleep",lambda seconds:None)
    def rss(*args,**kwargs):
        if fault=="observer":raise RuntimeError("observer unavailable")
        return (200 if fault=="rss" else 1),[{"pid":process.pid}]
    monkeypatch.setitem(sys.modules,"febio_runtime",SimpleNamespace(process_group_rss=rss))
    result=mesh.supervise_level(["not actually executed"],output,raw,p,.3 if fault=="wall" else 20)
    assert result["status"]=="failed" and result["automatic_retry"] is False
    assert result["kill_reason"]=={"wall":"wall_cap","rss":"process_family_rss_cap","active_output":"active_output_cap","total_output":"total_output_cap","observer":"supervision_error"}[fault]
    assert kills==[(process.pid,mesh.signal.SIGKILL)] and process.reaped
    assert json.loads((output/"supervision.json").read_text())["kill_reason"]==result["kill_reason"]
