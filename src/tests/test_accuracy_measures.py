import unittest

from src.tools.generic_accuracy import accuracy_funcs


class TestAccuracyMeasures(unittest.TestCase):

    def setUp(self):
        self.y_pred = {
            'correct_presence_only': [{'emotion': 'ang', 'salience': 1.0}],
            'incorrect_presence': [{'emotion': 'neu', 'salience': 1.0}],
            'correct_presence_salience': [{'emotion': 'disg', 'salience': 50}, {'emotion': 'hap', 'salience': 50}],
            'incorrect_salience': [{'emotion': 'disg', 'salience': 30}, {'emotion': 'hap', 'salience': 70}],
            'flipped_order_correct': [{'emotion': 'hap', 'salience': 70}, {'emotion': 'ang', 'salience': 30}],
            'wrong_salience_values': [{'emotion': 'ang', 'salience': 30}, {'emotion': 'hap', 'salience': 70}],
        }

        self.labels = {
            'correct_presence_only': [{'emotion': 'ang', 'salience': 1.0}],
            'incorrect_presence': [{'emotion': 'ang', 'salience': 1.0}],
            'correct_presence_salience': [{'emotion': 'hap', 'salience': 50}, {'emotion': 'disg', 'salience': 50}],
            'incorrect_salience': [{'emotion': 'hap', 'salience': 30}, {'emotion': 'disg', 'salience': 70}],
            'flipped_order_correct': [{'emotion': 'ang', 'salience': 30}, {'emotion': 'hap', 'salience': 70}],
            'wrong_salience_values': [{'emotion': 'ang', 'salience': 50}, {'emotion': 'hap', 'salience': 50}],
        }

    def test_acc_presence_single(self):
        self.assertTrue(accuracy_funcs.acc_presence_single(self.labels['correct_presence_only'], self.y_pred['correct_presence_only']))
        self.assertFalse(accuracy_funcs.acc_presence_single(self.labels['incorrect_presence'], self.y_pred['incorrect_presence']))
        self.assertTrue(accuracy_funcs.acc_presence_single(self.labels['correct_presence_salience'], self.y_pred['correct_presence_salience']))
        self.assertTrue(accuracy_funcs.acc_presence_single(self.labels['flipped_order_correct'], self.y_pred['flipped_order_correct']))
        self.assertTrue(accuracy_funcs.acc_presence_single(self.labels['wrong_salience_values'], self.y_pred['wrong_salience_values']))

    def test_acc_salience_single(self):
        self.assertTrue(accuracy_funcs.acc_salience_single(self.labels['correct_presence_salience'], self.y_pred['correct_presence_salience']))
        self.assertFalse(accuracy_funcs.acc_salience_single(self.labels['incorrect_salience'], self.y_pred['incorrect_salience']))
        self.assertTrue(accuracy_funcs.acc_salience_single(self.labels['flipped_order_correct'], self.y_pred['flipped_order_correct']))
        self.assertFalse(accuracy_funcs.acc_salience_single(self.labels['wrong_salience_values'], self.y_pred['wrong_salience_values']))

    def test_acc_salience_single_raises(self):
        # Length mismatch
        with self.assertRaises(ValueError):
            accuracy_funcs.acc_salience_single(self.labels['correct_presence_only'], self.y_pred['correct_presence_only'])


if __name__ == '__main__':
    unittest.main()
