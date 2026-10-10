"""Conservative metadata-only classification; no transport or filesystem mutation."""
def classify(entry, history, orphan_intents, final_exists, partial_regular, partial_bytes):
    if orphan_intents:
        return 'exclude_in_flight_or_unreconciled_intent'
    if final_exists:
        return 'exclude_local_final_requires_verification'
    if not partial_regular or not 0 <= partial_bytes <= entry['bytes']:
        return 'exclude_partial_type_or_size'
    if not history:
        return 'exclude_no_history'
    def known_transport(row):
        kind=row.get('error_type');error=row.get('error','')
        return ((kind=='URLError' and error in ['<urlopen error [Errno 8] nodename nor servname provided, or not known>','<urlopen error [Errno 54] Connection reset by peer>'])
                or (kind=='TimeoutError' and error=='The read operation timed out')
                or (kind=='BrokenPipeError' and error=='[Errno 32] Broken pipe'))
    if any(not known_transport(h['result']) for h in history):
        return 'exclude_other_error_or_integrity_evidence'
    last=history[-1];r=last['result']
    if r['status']=='transport_exhausted' and r['error_type'] in ['URLError','TimeoutError']:
        return 'proposed_exhausted_transport_budget_extension'
    if r['status']=='integrity_or_scope_refusal' and r['error_type']=='BrokenPipeError':
        h=last.get('http');offset=last['intent']['resume_offset']
        if not h:
            return 'exclude_broken_pipe_without_http_stage_evidence'
        headers=h.get('headers',{})
        expected_range=f"bytes {offset}-{entry['bytes']-1}/{entry['bytes']}" if offset else None
        if not (h.get('status')==(206 if offset else 200) and h.get('effective_url')==entry['source_url']
                and headers.get('Content-Length')==str(entry['bytes']-offset)
                and headers.get('Content-Range')==expected_range
                and headers.get('ETag')==entry['etag_opaque']
                and headers.get('x-amz-version-id')==entry['s3_version_id']
                and headers.get('Content-Encoding') in [None,'identity']):
            return 'exclude_broken_pipe_http_identity_mismatch'
        if partial_bytes >= entry['bytes']:
            return 'exclude_full_partial_requires_local_verification'
        return 'proposed_worker_broken_pipe_transport_reclassification'
    return 'exclude_nonterminal_or_other_status'
