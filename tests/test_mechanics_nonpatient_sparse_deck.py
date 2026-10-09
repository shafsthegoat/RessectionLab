"""Source-only structured tet10 and XML checks; no FEBio or patient reads."""
from dataclasses import replace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_patient_constraints as fixture


def test_one_cube_control_matches_pinned_fixture_by_physical_node_and_element():
    mesh = deck.build_mesh(1, software_control=True)
    X, E, _ = fixture.fixture_mesh()
    lookup = {tuple(np.round(x, 15)): i for i,x in enumerate(mesh.nodes_m)}
    remapped = np.array([[lookup[tuple(np.round(X[node], 15))] for node in row] for row in E])
    assert np.array_equal(remapped, mesh.tet10_indices)
    assert deck.validate_mesh(mesh)["shared_edges"] == 19


@pytest.mark.parametrize("n,nodes,tets,edges,boundary", [
    (5,1331,750,1115,602),
    (9,6859,4374,5859,1946),
    (13,19683,13182,16939,4058),
])
def test_three_frozen_levels_have_actual_conforming_counts_and_volume(n,nodes,tets,edges,boundary):
    mesh = deck.build_mesh(n)
    result = deck.validate_mesh(mesh)
    assert (result["nodes"],result["tet10"],result["shared_edges"],result["boundary_nodes"]) == (
        nodes,tets,edges,boundary)
    assert result["reference_samples_per_element"] == 19
    assert result["total_reference_volume_m3"] == pytest.approx(1e-6,abs=1e-18)
    assert result["minimum_reference_det_m3"] == pytest.approx((.01/n)**3,rel=1e-12)
    assert len(mesh.face_node_ids["top"]) == (2*n+1)**2
    assert len(mesh.face_node_ids["bottom"]) == (2*n+1)**2


def test_geometry_witness_rejects_bad_midnode_reference_and_boundary_tag():
    mesh = deck.build_mesh(5)
    wrong = mesh.tet10_indices.copy()
    wrong[0,4] = wrong[0,5]
    with pytest.raises(ValueError):
        deck.validate_mesh(replace(mesh,tet10_indices=wrong))
    wrong = mesh.tet10_indices.copy()
    wrong[0,4] = wrong[1,4]
    with pytest.raises(ValueError):
        deck.validate_mesh(replace(mesh,tet10_indices=wrong))
    bad = dict(mesh.face_node_ids)
    bad["top"] = bad["top"][:-1]
    with pytest.raises(ValueError,match="boundary tag"):
        deck.validate_mesh(replace(mesh,face_node_ids=bad))
    displaced = mesh.nodes_m.copy()
    displaced[mesh.tet10_indices[0,4],0] += 1e-5
    with pytest.raises(ValueError):
        deck.validate_mesh(replace(mesh,nodes_m=displaced))
    with pytest.raises(ValueError):
        deck.build_mesh(2)


@pytest.mark.parametrize("case_id", design.CASE_ORDER)
def test_exact_deck_identity_and_boundary_protocol(case_id):
    xml, report = deck.build_deck(case_id)
    if case_id == "n5_affine":
        assert deck.build_deck(case_id)[0] == xml
    root = ET.fromstring(xml)
    declared = design.validate_declaration()
    case = next(row for row in declared["cases"] if row["id"] == case_id)
    n = case["n"]
    mesh = deck.build_mesh(n)
    assert len(root.findall("Mesh/Nodes/node")) == (2*n+1)**3
    assert len(root.findall("Mesh/Elements/elem")) == 6*n**3
    assert root.find("Mesh/Elements").attrib["type"] == "tet10"
    domain = root.find("MeshDomains/SolidDomain")
    assert domain.attrib == {"name":"cube","mat":"numerical_ogden",
                             "elem_type":"TET10G8","type":"elastic-solid"}
    material = root.find("Material/material")
    assert material.attrib["type"] == "Ogden"
    assert float(material.findtext("c1")) == 2000.
    assert float(material.findtext("k")) == pytest.approx(1000*29/3)
    assert material.findtext("pressure_model") == "1"
    assert [float(material.findtext(f"c{i}")) for i in range(2,7)] == [0.]*5
    solver = root.find("Control/solver")
    assert solver.attrib["type"] == "solid"
    assert solver.find("qn_method").attrib["type"] == "BFGS"
    assert solver.find("linear_solver").attrib["type"] == "accelerate"
    assert {node.tag:node.text for node in solver.find("linear_solver")} == {
        "iterative":"0","factorization":"4","order_method":"0","print_condition_number":"0"}
    for field,value in (("symmetric_stiffness",1),("max_refs",20),("dtol",1e-10),
                        ("etol",1e-10),("rtol",1e-10),("min_residual",1e-20)):
        assert float(solver.findtext(field)) == value
    steps = case.get("time_steps",4)
    dt = case.get("step_size",.25)
    assert int(root.findtext("Control/time_steps")) == steps
    assert float(root.findtext("Control/step_size")) == dt
    assert root.findtext("Control/time_stepper/max_retries") == "0"
    assert float(root.findtext("Control/time_stepper/dtmin")) == dt
    assert float(root.findtext("Control/time_stepper/dtmax")) == dt
    assert root.find("Output/logfile/node_data").attrib["data"] == fixture.patch.NODE_FIELDS
    assert root.find("Output/logfile/element_data").attrib["data"] == fixture.patch.ELEMENT_FIELDS
    assert root.findtext("LoadData/load_controller/points/pt") == "0,0"
    sets = {x.attrib["name"]:{int(y)-1 for y in x.text.split(",")}
            for x in root.findall("Mesh/NodeSet")}
    bcs = root.findall("Boundary/bc")
    if case["load"] == "affine":
        exterior = set(np.concatenate(list(mesh.face_node_ids.values())))
        assert len(sets) == len(exterior)
        assert set.union(*sets.values()) == exterior
        assert len(bcs) == 3*len(exterior)
        F = np.array(declared["loads"]["affine"]["final_F"])
        for bc in bcs:
            index = next(iter(sets[bc.attrib["node_set"]]))
            axis = "xyz".index(bc.findtext("dof"))
            target = ((F-np.eye(3)) @ mesh.nodes_m[index])[axis]
            assert float(bc.findtext("value")) == pytest.approx(target,abs=1e-17)
    else:
        assert sets == {"bottom":set(mesh.face_node_ids["bottom"]),
                        "top":set(mesh.face_node_ids["top"])}
        assert len(bcs) == 6
        for bc in bcs:
            axis = "xyz".index(bc.findtext("dof"))
            target = declared["loads"]["nonuniform"][bc.attrib["node_set"] + "_displacement_m"][axis]
            assert float(bc.findtext("value")) == target
    assert report["scope"].startswith("In-memory source-only")


def test_undeclared_case_cannot_generate_deck():
    with pytest.raises(ValueError,match="Undeclared"):
        deck.build_deck("n17_affine")
