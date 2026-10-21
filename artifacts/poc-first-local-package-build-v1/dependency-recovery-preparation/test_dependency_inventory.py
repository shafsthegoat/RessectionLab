"""Generated metadata controls only; no engine, numerical package or patient access."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('closure', Path(__file__).with_name('audit_dependency_closure.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def dist(name, version, *paths):
    return {'name': name, 'version': version,
        'record_files': [{'record_path': p, 'source': p} for p in paths]}


class OwnershipControls(unittest.TestCase):
    def test_record_ownership_handles_import_alias_and_vendored_module(self):
        included = {'PIL/Image.py': ['entry'], 'setuptools/_vendor/packaging/__init__.py': ['entry']}
        result = audit.ownership(included, [dist('Pillow', '1', 'PIL/Image.py'),
            dist('setuptools', '2', 'setuptools/_vendor/packaging/__init__.py'),
            dist('packaging', '3', 'packaging/__init__.py')], {'pillow': '1', 'setuptools': '2'})
        self.assertEqual([x['name'] for x in result['distributions']], ['Pillow', 'setuptools'])
        self.assertFalse(result['missing_pins'] or result['version_mismatches'] or result['unowned'])

    def test_all_missing_and_mismatched_distributions_are_reported(self):
        result = audit.ownership({'a': [], 'b': [], 'c': []},
            [dist('A', '1', 'a'), dist('B', '2', 'b'), dist('C', '3', 'c')], {'b': '1'})
        self.assertEqual([x['name'] for x in result['missing_pins']], ['A', 'C'])
        self.assertEqual(result['version_mismatches'], [{'name': 'B', 'installed_version': '2', 'locked_version': '1'}])

    def test_corrected_pin_resolves_only_the_missing_pin(self):
        ds = [dist('psutil', '7.2.2', 'psutil/__init__.py')]
        before = audit.ownership({'psutil/__init__.py': []}, ds, {})
        after = audit.ownership({'psutil/__init__.py': []}, ds, audit.locks('psutil==7.2.2\n'))
        self.assertEqual(len(before['missing_pins']), 1)
        self.assertEqual(after['missing_pins'], [])
        self.assertEqual(after['version_mismatches'], [])

    def test_unknown_and_ambiguous_owners_are_not_silently_assigned(self):
        result = audit.ownership({'one': [], 'unknown': []}, [dist('a', '1', 'one'), dist('b', '1', 'one')], {'a': '1', 'b': '1'})
        self.assertEqual(result['unowned'], ['unknown'])
        self.assertEqual(result['ambiguous_record_owners'], {'one': ['a', 'b']})

    def test_canonical_names_and_duplicate_lock_refusal(self):
        self.assertEqual(audit.locks('Lazy_Loader==0.6\n'), {'lazy-loader': '0.6'})
        with self.assertRaises(ValueError):
            audit.locks('Lazy_Loader==0.6\nlazy-loader==0.6\n')

    def test_nested_toc_extracts_only_triple_string_records(self):
        self.assertEqual(list(audit.rows([('a', '/source/a.py', 'PYMODULE'),
            [17, ('b', '/source/b.so', 'EXTENSION')], ('incomplete', 'row')])),
            [('a', '/source/a.py', 'PYMODULE'), ('b', '/source/b.so', 'EXTENSION')])


if __name__ == '__main__':
    unittest.main()
