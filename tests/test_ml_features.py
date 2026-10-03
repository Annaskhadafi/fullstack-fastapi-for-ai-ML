import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import ml_service, model_hub_service


class ModelFeaturesTest(unittest.TestCase):
    def test_database_schema_reaches_dynamic_form(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            path.touch()
            model = SimpleNamespace(id='test', name='test', display_name='Test', description='',
                                    category='machine_learning', framework='scikit-learn', task_type='regression',
                                    file_path=str(path), created_at='', features_json='[{"name":"age","label":"Umur","default":25}]',
                                    target_names_json='[]', metrics_json='{}')
            db = AsyncMock()
            db.execute.return_value.scalars = lambda: SimpleNamespace(all=lambda: [model])
            with patch.object(model_hub_service, 'scan_weights_directory', return_value=[]), patch.object(model_hub_service, 'load_local_registry', return_value=[]):
                models = asyncio.run(ml_service.list_available_models(db))
            self.assertEqual(models[0].features[0].name, 'age')
            self.assertEqual(models[0].features[0].label, 'Umur')
            self.assertEqual(models[0].features[0].default, 25)

    def test_named_and_unnamed_model_inputs(self):
        named = SimpleNamespace(feature_names_in_=['age', 'income'])
        self.assertEqual([f.name for f in ml_service._inferred_features(named, 'test')], ['age', 'income'])
        unnamed = SimpleNamespace(n_features_in_=3)
        self.assertEqual(len(ml_service._inferred_features(unnamed, 'test')), 3)


if __name__ == '__main__':
    unittest.main()
