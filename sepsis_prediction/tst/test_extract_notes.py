import unittest
from features.extract_notes import get_cohort_trimmed, process_text

class ExtractNotesTestCase(unittest.TestCase):

    def test_process_text(self):
        input = """
        Admission Date:  [**2151-7-16**]       Discharge Date:  [**2151-8-4**]


        Service:
        ADDENDUM:
        
        RADIOLOGIC STUDIES:  Radiologic studies also included a chest
        CT, which confirmed cavitary lesions in the left lung apex
        consistent with infectious process/tuberculosis.  This also
        moderate-sized left pleural effusion.
        
        HEAD CT:  Head CT showed no intracranial hemorrhage or mass
        effect, but old infarction consistent with past medical
        history.
        
        ABDOMINAL CT:  Abdominal CT showed lesions of
        T10 and sacrum most likely secondary to osteoporosis. These can
        be followed by repeat imaging as an outpatient.
        
        
        
        [**First Name8 (NamePattern2) **] [**First Name4 (NamePattern1) 1775**] [**Last Name (NamePattern1) **], M.D.  [**MD Number(1) 1776**]
        
        Dictated By:[**Hospital 1807**]
        MEDQUIST36
        
        D:  [**2151-8-5**]  12:11
        T:  [**2151-8-5**]  12:21
        JOB#:  [**Job Number 1808**]
"""

        expected = 'radiolog studi also includ chest confirm cavitari lesion left lung apex consist infecti process tuberculosi also moder size left pleural effus head show intracrani hemorrhag mass effect old infarct consist past medic histori abdomin show lesion t10 sacrum like secondari osteoporosi follow repeat imag outpati'

        actual = process_text(input)
        self.assertEqual(expected, actual)

    def test_cohort_trimmed(self):
        df = get_cohort_trimmed()
        self.assertEqual(1767, df.shape[0])
        print(df.head())