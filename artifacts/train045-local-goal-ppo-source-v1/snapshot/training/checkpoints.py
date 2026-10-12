"""Lossless standard-SB3 ZIP transport; never persist the last image observation."""
import hashlib,io,zipfile


def checkpoint_bytes(model, *, decoded_cap=64*1024**2, compressed_cap=2*1024**2):
    original=io.BytesIO()
    model.save(original,exclude=['_last_obs','_last_original_obs'])
    if original.tell()>decoded_cap:raise ValueError('Decoded SB3 checkpoint cap')
    original.seek(0);packed=io.BytesIO();members={}
    with zipfile.ZipFile(original,'r') as src, zipfile.ZipFile(packed,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as dst:
        names=src.namelist()
        if len(names)!=len(set(names)) or 'policy.optimizer.pth' not in names:raise ValueError('Complete unique optimizer checkpoint required')
        for name in names:
            raw=src.read(name);members[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            dst.writestr(info,raw,compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    raw=packed.getvalue()
    if len(raw)>compressed_cap:raise ValueError('Compressed SB3 checkpoint cap')
    return raw,{'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
        'encoding':'standard_SB3_ZIP_losslessly_DEFLATED','members':members,
        'decoded_member_bytes':sum(x['bytes'] for x in members.values()),
        'last_image_observations_excluded':True,'includes_optimizer_state':True}
