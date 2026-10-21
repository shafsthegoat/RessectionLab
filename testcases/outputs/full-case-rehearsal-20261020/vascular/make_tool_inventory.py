from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
sys.path.insert(0,'src')
from resectionlab.surgical_tools import TOOLS
from resectionlab.surgical_environment import load_environment
base=Path('testcases/outputs/full-case-rehearsal-20261020/vascular')
scene=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
cases=ET.parse(base/'tool-family-fixture.xml').getroot().findall('.//testcase')
materials={r['material'] for r in scene.regions}
rows=[]
limits={'dural_scissors':'missing-anatomy: no membrane supplied',
'periosteal_elevator':'missing-attachment-evidence: no declared periosteal bonds',
'dissector':'missing-attachment-evidence: no declared soft bonds',
'craniotome':'component separability unknown: no declared bone anchors',
'cortical_probe':'mapping remains unknown: no supplied functionalEvidence',
'subcortical_probe':'mapping remains unknown: no supplied functionalEvidence',
'brain_retractor':'model supported; patient-specific access requires checked geometry',
'tumor_forceps':'model supported; extraction requires proven free fragment'}
for tool,data in TOOLS.items():
    matches=[c for c in cases if '['+tool+'-' in c.attrib['name']]
    assert len(matches)==1,tool
    test=matches[0];status='passed' if not list(test) else 'failed'
    rows.append({'tool':tool,'effect':data['effect'],'fixtureStatus':status,'fixtureTest':test.attrib['name'],'fixtureWallSeconds':float(test.attrib['time']),'patientStatus':limits.get(tool,'model supported; actual patient action still requires checked geometry'),'patientProof':'patient action receipts cover drill, CUSA, scissors, bipolar, irrigation, suction in combined prefix/rescue; other tools retain fixture-only proof'})
report={'schema':'resectionlab.tool-family-checks.v1','sceneId':scene.identity,'fixture':'tests/test_surgical_tools.py::test_engine_step_instrument_admissibility','status':'passed' if all(x['fixtureStatus']=='passed' for x in rows) else 'failed','toolCount':len(rows),'tools':rows,'limits':['Synthetic engineering fixture is explicitly separate from patient anatomy.','Fixture suction test declares an initial local fluid volume directly; patient rescue uses generated bleeding and public irrigation instead.','No membrane, attachment, functional response or bone-anchor evidence was fabricated for the patient.']}
(base/'tool-family-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],len(rows))
