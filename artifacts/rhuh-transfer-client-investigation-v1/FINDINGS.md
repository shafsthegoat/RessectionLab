# RHUH public transfer client investigation

The concrete route is IBM **Transfer SDK 1.1.9 for macOS arm64**, listed in the
[official pinned SDK catalog](https://github.com/IBM/aspera-cli/blob/5676d597ec90be0b3c94efba5d12a35b77f5a900/docs/sdk_location.yaml).
Its 25,582,806-byte ZIP was retrieved anonymously to ignored
`build/rhuh-transfer-client-inspection-v1/`; no installer or packaged executable
ran. Local SHA256 is
`5101c4cff354b5021b57ca28bd48894254f237b6894e4c9967a0e15a40eda6fa`.
The publisher supplied no SHA256 in the catalog/HEAD response; the multipart
ETag is not treated as a content checksum. All 132 ZIP member CRCs passed.

Static inspection identifies `ascp` as native arm64, minimum macOS 11.0, with
only system-library links. Strict code-signature verification passed for
**Developer ID Application: Aspera, Inc. (RJ747GSBCT)**. This establishes signed
binary integrity; runtime compatibility on this arm64 macOS 26.6 host remains
untested. The extracted inspection copies remain mode 0644.

[IBM's developer FAQ](https://www.ibm.com/products/aspera/support) says SDK use
has no additional developer cost; premium support is optional. The package
includes an opaque runtime `aspera-license`. The CLI wrapper is Apache-2.0;
that does not establish the SDK binary's redistribution terms. No product
bundling or redistribution is proposed. The
[IBM CLI manual](https://github.com/IBM/aspera-cli/blob/5676d597ec90be0b3c94efba5d12a35b77f5a900/docs/README.md#fasp-protocol-ascp)
documents direct `ascp` use and isolated SDK extraction.

The public Connect download currently points to an x86_64 package. The native
SDK avoids that architecture dependency; the current CLI release provides no
macOS standalone asset, so installing Ruby/gems is unnecessary for the first
check.

After root review, the precise next check is to add owner-execute permission
to the isolated binary and run it with **`-A` only**, the version flag used by
[IBM's implementation](https://github.com/IBM/aspera-cli/blob/5676d597ec90be0b3c94efba5d12a35b77f5a900/lib/aspera/ascp/installation.rb#L169).
`next-step.json` contains the exact absolute path and argument arrays. No
system installation, administrator privileges, network destination or transfer
credentials are needed for this proposed check. A checksum-only RHUH transfer
remains a separate review step; patient images remain unreleased.
