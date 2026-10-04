# Release evidence audit

Static inspection, October 4, 2026. No build, test, download, app launch, model
execution or patient processing was performed. This records observed license
material and gaps; it does not select a project license or grant new rights.

## Inspected release and scope

The current Electron app is
`desktop/release/RessectionLab-darwin-arm64/RessectionLab.app`, renderer capture
`762973e28edb5d2697618a08d2332674dad170ff4f2d2d8ad625313bb642d831`.
Its numerical executable SHA-256 is
`85a92eff69cb1ea97afc400bc80433c53d146174dd32a9cf36061b8420f72ddd`.
The embedded Python source manifest and existing build report agree on revision
`0b4e3345137316bb72d6daf9bf2a15ee6b613ed7`, source digest `afd24365…` and
snapshot `build/electron-sidecar/inputs/20261004T103548.141696Z`. The existing
`Analysis-00.toc` names that same snapshot. See [packaged execution evidence](electron-packaging.md).

The app starts empty and can generate its own synthetic fixture. The packaging
recipe does not copy the ignored `data/` or `outputs/` directories; Vite disables
public-directory copying. Static inspection found no MRI volume, case bundle or
trained checkpoint in this app. The two shipped `.npy` files are scikit-image
morphology lookup tables. Imported patient cases, prior proposals and research
models have separate provenance and do not become redistributable just because
the software can load them. The signed artifact remains ad hoc and not notarized;
current research learners must not inherit this older engine's validation.

## Concrete synthetic-demo release gaps

| Finding in the inspected artifact | Existing evidence / action |
|---|---|
| Project release terms are undeclared. There is no root `LICENSE` or `THIRD_PARTY_NOTICES.md`; `pyproject.toml` and `desktop/package.json` lack a license declaration. | An owner must choose the intended first-party release terms. Do not label the repository open source or promise downstream redistribution rights from its public visibility. Do not substitute a dependency's MIT license. |
| The renderer bundles React, React DOM, Three.js, Radix UI and Lucide, but its generated JavaScript retains no license/copyright comments and `app.asar` has no notice file. | Exact installed `LICENSE` files exist under `desktop/node_modules/`. Capture the actual renderer dependency closure, its versions and complete upstream texts. Lucide's full file includes more than a one-word ISC label. |
| Electron's `LICENSE` and `LICENSES.chromium.html` exist **beside** the `.app`, not inside it. | A distribution containing only the `.app` drops those files. Include the exact locked Electron/Chromium texts inside the app and verify the final artifact. |
| The frozen engine contains 114 license-like files, all attributable to NumPy, Torch, tqdm and MarkupSafe. Other bundled libraries lack their corresponding notice material. | Add exact installed SciPy, nibabel, scikit-image, Pillow, h5py, CPython and transitive-package notices. Preserve vendor texts rather than reducing each package to its headline license identifier. |
| The available CPython runtime has its own license, but no matching aggregate vendor-license/source manifest was found. Its recorded build configuration includes statically linked dependencies. | Obtain the exact runtime's vendor provenance and applicable notices, or use a runtime with an auditable distribution record. Copying CPython's license alone does not establish completeness for OpenSSL, expat, libffi, SQLite, mpdecimal, readline/ncurses and other linked components; build flags alone do not establish exact vendor versions or terms. |
| Historical Qt documentation says the bundle retains dependency notices. That is not evidence for the Electron recipe. | `packaging/RessectionLab.spec` has explicit notice collection; `scripts/build_electron_sidecar.py` and `desktop/electron/package-macos.cjs` did not have equivalent complete collection at this inspection. Record any source fix separately from a rebuilt and inspected package. |

The packaging agent is implementing deterministic notice capture during the
heavy-work hold. **No newly compliant bundle is claimed by this audit.** Inclusion
must be tied to the same captured inputs, package versions and executable that
will be distributed; a file added to the checkout alone does not repair an
existing app.

## Exact local notice inputs for the packaging follow-up

Installed Python distribution records identify these currently missing inputs:

- `scipy-1.18.1.dist-info/LICENSE.txt`, plus package-local uarray, duccfft,
  DOP and Qhull notices.
- `nibabel-5.4.2.dist-info/licenses/COPYING` and
  `scikit_image-0.26.0.dist-info/LICENSE.txt`.
- `pillow-12.3.0.dist-info/licenses/LICENSE`, which also contains vendor material;
  h5py's `licenses/` tree, including HDF5 and LZF notices.
- The build runtime's `lib/python3.12/LICENSE.txt`. PyInstaller's
  `COPYING.txt` and its bundling exception are distinct from CPython's terms.

The matched freezer analysis also contains setuptools and vendored modules,
typing_extensions, packaging, sympy, mpmath, filelock, fsspec, networkx, Jinja2,
rich, markdown-it-py, mdurl and lazy-loader. It contains `_pytest`, iniconfig,
Pygments, pluggy and matplotlib-related module entries despite some top-level
exclusions. Use actual collected module origins and distribution records to
capture their notices; do not assume that development-looking names are absent.
Native libraries include imaging codecs, HDF5 and OpenMP, requiring coverage by
their originating wheel/runtime notices. This is an inventory requirement, not
a claim that every optional feature of those packages is present.

## Optional research assets: separate rights and evidence

| Asset | What is already recorded | Remaining boundary |
|---|---|---|
| UCSF structural mirror | [Acquisition manifest](../manifests/data_acquisition_ucsf.json) records CC BY 4.0, authors, dataset/article DOIs, mirror revision and unchanged downloaded bytes. | Official-release byte equivalence and mirror processing remain unverified. Any redistributed derivative figure/case needs its source attribution and transformation description; it is not part of the synthetic app. |
| BTC PAT28/PAT05 | [Creator-source record](diffusion_source.md) and acquisition manifests retain CC0, Aerts/Marinazzo, DOI/version and requested OpenNeuro acknowledgment. | Article terms are separate from dataset terms. Keep identities and transformation histories with released research artifacts. |
| Seven motor/language prior maps | [Functional evidence](functional_evidence.md) records Zenodo CC BY 4.0, exact members/hashes and source papers. | [Exact embedded MNI152 template redistribution remains unresolved](prior-registration.md). Record-level terms must not be silently extended to upstream template bytes or reproduced article figures. |
| SynthStrip | [Model manifest](synthstrip-model-manifest.json), saved official page and [FreeSurfer license copy](licenses/FreeSurfer-SynthStrip.txt) distinguish weights offered under MIT/CC BY 4.0 from code under FreeSurfer terms. Modified-device adapter and outputs are identified in [the experiment](brain_extraction.md). | Current app ships no weights/upstream executable. If shipping this optional path later, carry the full applicable code terms, required attribution, modification notice and the chosen weight notice. The saved offer is not a blanket license for FreeSurfer or its training datasets. |
| PyHySCO | [Isolated phantom experiment](pyhysco-phantom.md) records GPL-3.0 wheel hash and private optional runtime. | The optional package is outside app dependencies and payload. No GPL redistribution arrangement or patient correction is inferred from this private experiment. |
| FSL/TORTOISE | [Dependency audit](diffusion-correction-options.md) records the unresolved FSL-use question and TORTOISE's bundled BET linkage. | Neither is included by the inspected app; do not add it under TORTOISE's headline GPL label alone. |

Existing data citations are usable starting points, but a public screenshot or
paper-figure export should carry an adjacent source/derivative credit rather
than relying on an unrelated acquisition document being discovered. The current
UI icon is documented as an original project vector; the renderer uses system
fonts and no downloaded Medivis visual asset was found in its packaging inputs.

Scientific citations, dataset licenses, model terms, first-party release terms
and clinical evidence are separate records. This audit changes none of the
existing research-only claims or evidence-acceptance gates.

**Audit frozen:** the observations above apply to the identified existing
artifact. Keep them as historical findings. A later source-inventory review or
rebuilt-package inspection should be recorded as an addendum with its own exact
artifact identity; it must not retroactively mark this package complete.

## Addendum: notice-source inventory review, October 4, 2026

Static review of packaging's first local collector outputs, before integration
into a rebuilt app:

- `build/dependency-notices/python-probe-v1/inventory.json`, SHA-256
  `76312cbe4557f5e07e38c23c593a9c0b0d7cd4e1505966daaadeddf81904b225`, identifies
  the same numerical executable above, 28 distribution owners and 225 copied
  upstream files. Its recorded archive check covers 5,428 module names and
  5,420 matching source code objects. The inventory retains package/vendor
  notice trees, including the previously missed pytest, matplotlib and
  setuptools dependencies. The file count includes metadata and incidental
  `packaging/licenses/` Python files; it is not a count of distinct licenses.
- `build/dependency-notices/renderer-probe-v1/renderer-inventory.json`, SHA-256
  `f053dadfc21e673313373758e265bafd8a5658a683780f055a05a1f462501233`, records
  35 packages and 69 metadata/notice files, matching an independent traversal
  of installed production dependencies and runtime peers. Exact Electron and
  Chromium texts were also captured alongside this probe.

The captured renderer explicitly leaves `react-remove-scroll-bar@2.3.8`
unresolved: its installed package declares MIT, but contains no license/notice
file, and its README provides only that identifier. Obtain authoritative text
for that version; another package's copyright notice is not a substitute.
The CPython static-vendor provenance gap remains explicit. First-party release
terms and template redistribution rights are unchanged. This source-inventory
review does not certify native-binary provenance, final app inclusion, or
complete redistribution rights; those require separate evidence for the final
artifact.
