"""Only the separately authenticated release linkage may differ."""
import copy,hashlib,json,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import semantic
from admission_parity import admission_join


def pair():
    accounts=[];chains=[]
    for index in ('old-index','new-index'):
        release={'root_source_index_sha256':index,'scope':'same','caps':{'same':1}}
        raw=json.dumps(release,sort_keys=True).encode();digest=hashlib.sha256(raw).hexdigest()
        protocol={'runtime_release_sha256':digest,'world':'same'}
        original={'protocol_sha256':semantic(protocol),'anatomy':'same'}
        context={'original_context_hash':semantic(original),'catalog':'same','frontier':'same'}
        accounts.append({'context':context,'context_seal':semantic(context)})
        chains.append({'context':original,'bindings':{'protocol':protocol,'source_qc':'same'},'release':release,
            'release_sha256':digest,'source_index_sha256':index})
    return accounts,chains


def test_different_release_join_is_allowed_with_every_scientific_field_exact():
    a,c=pair();result=admission_join(*a,*c)
    assert result['old_context_seal']!=result['new_context_seal']
    assert result['old_original_context_hash']!=result['new_original_context_hash']


@pytest.mark.parametrize('what',['foreign_context','foreign_release','physics','protocol','world','source_qc'])
def test_unjoined_or_scientific_changes_refused(what):
    a,c=pair()
    if what=='foreign_context':a[1]['context']['original_context_hash']=a[0]['context']['original_context_hash']
    elif what=='foreign_release':c[1]['bindings']['protocol']['runtime_release_sha256']='foreign'
    elif what=='physics':a[1]['context']['catalog']='different'
    elif what=='protocol':c[1]['bindings']['protocol']['world']='different'
    elif what=='world':c[1]['context']['anatomy']='different'
    else:c[1]['bindings']['source_qc']='different'
    with pytest.raises(ValueError):admission_join(*a,*c)
