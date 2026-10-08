# Exact-source native trust correction

The frozen NIRD original remains available. Python lacks the HARICA root already
trusted by macOS. The [diagnosis](diagnosis.json) includes independent macOS
hostname/certificate verification and a system-curl HTTP200 HEAD with the exact
9,156,131-byte length. No trust anchor was added and TLS was not disabled.

Only this exact original-image entry uses `/usr/bin/curl -q` with the existing
macOS trust store. Source size/MD5, runtime identity, original URL, exclusive
publication, deadline and process cleanup remain enforced. OSF transport stays
unchanged. No automatic retries or redirects are enabled for this image.

Owner42controls passed in1.21seconds; [independent review](independent-review.json)
passed32relevant controls in0.98seconds and separately verified deadline cleanup
with a real local sleeping process. No network or scientific payload was used
for these controls. The earlier Python certificate failure remains preserved.

Next: one bounded original/mask acquisition, followed by separate structural QC.
