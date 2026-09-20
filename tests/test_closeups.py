import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import yaml

from src.common import CLASS_NAMES
from src.prepare_closeups import closeup_geometry, prepare_closeups
from src.preprocessing import read_labels


class CloseupTests(unittest.TestCase):
    def test_known_box_and_class_preservation(self):
        bounds, boxes = closeup_geometry([(1, .5, .5, .2, .4)], (100, 100), padding=0)
        self.assertEqual(bounds, (40, 30, 60, 70))
        self.assertEqual(boxes, [(1, .5, .5, 1., 1.)])

    def test_union_preserves_every_object_and_edges(self):
        source = [(1, .1, .1, .2, .2), (4, .8, .8, .2, .2)]
        bounds, boxes = closeup_geometry(source, (100, 100))
        self.assertEqual(bounds, (0, 0, 100, 100))
        for original, converted in zip(source, boxes):
            self.assertEqual(original[0], converted[0])
            for a, b in zip(original[1:], converted[1:]):
                self.assertAlmostEqual(a, b)

    def test_invalid_and_empty_inputs(self):
        self.assertIsNone(closeup_geometry([], (100, 100)))
        for padding in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                closeup_geometry([], (100, 100), padding)
        with self.assertRaises(ValueError):
            closeup_geometry([(1, .5, .5, 2, 1)], (100, 100))

    def test_separate_dataset_keeps_source_and_validation_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            for index, split in enumerate(('train', 'val', 'test')):
                (source / 'images' / split).mkdir(parents=True)
                (source / 'labels' / split).mkdir(parents=True)
                Image.new('RGB', (100, 100), (index*40, 30, 60)).save(source / 'images' / split / 'image.png')
                (source / 'labels' / split / 'image.txt').write_text('1 0.5 0.5 0.2 0.4\n')
            config = source / 'dataset.yaml'
            config.write_text(yaml.safe_dump({'path': str(source), 'names': dict(enumerate(CLASS_NAMES)),
                                              **{s: f'images/{s}' for s in ('train','val','test')}}))
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
            destination = root / 'derived'
            report = prepare_closeups(config, destination)
            self.assertEqual(report['crops_added'], 1)
            self.assertEqual(report['after']['train']['images'], 2)
            self.assertEqual(len(read_labels(destination / 'labels/train/closeup__image.txt')), 1)
            for split in ('val', 'test'):
                for kind, ext in (('images','png'), ('labels','txt')):
                    relative = Path(kind) / split / f'image.{ext}'
                    self.assertEqual((source / relative).read_bytes(), (destination / relative).read_bytes())
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):
                prepare_closeups(config, destination)


if __name__ == '__main__':
    unittest.main()
