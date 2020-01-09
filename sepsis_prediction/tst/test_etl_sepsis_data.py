import unittest
import pandas as pd
import sys
import logging
from features.etl_sepsis_data import create_labels, import_csv, preprocess_data

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class MyTestCase(unittest.TestCase):
    def test_import_csv(self):
        csv = import_csv()
        self.assertEqual((34173, 87), csv.shape)

    def test_create_labels(self):
        test_data = {'CaseControl':['Case', 'Control'], 'subject_id':['patientA', 'patientB']}
        df = pd.DataFrame(test_data)
        labels = create_labels(df)
        self.assertEqual(1, labels['patientA'])
        self.assertEqual(0, labels['patientB'])

    def test_preprocess_data(self):
        df = pd.DataFrame({
            'Pred or Obs': ['Observation', 'Prediction', 'Prediction','Prediction'],
            'index_hrs': [1, 13, 5, 4],
            'admission_age': [73, 56, 15, 16]
        })
        df = preprocess_data(df)
        self.assertEqual(1, df.shape[0])

if __name__ == '__main__':
    unittest.main()
