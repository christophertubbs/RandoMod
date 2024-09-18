import typing
import unittest

from randomod import model


class ModelHelpersTest(unittest.TestCase):
    def test_get_threshold_for_value(self):
        def get_test_value(index: int) -> float:
            return 14.0 + 5.0 * index

        thresholds: typing.Dict[str, float] = {
            "threshold_1": 15,
            "threshold_2": 20,
            "threshold_3": 25,
            "threshold_4": 30,
        }
        threshold_names: typing.List[str] = list(thresholds.keys())
        for input_index in range(6):
            value = get_test_value(input_index)
            threshold_index, threshold_name, maximum_value = model.get_threshold_for_value(
                value,
                thresholds
            )

            if threshold_index == 0:
                self.assertEqual(threshold_name, threshold_names[input_index])
                self.assertLess(value, maximum_value)
            elif threshold_index >= len(thresholds):
                self.assertEqual(threshold_name, model.OVERFLOW_THRESHOLD_NAME)
                _, _, calculated_maximum = model.get_overflow_threshold(
                    threshold_index,
                    value,
                    thresholds[threshold_names[-1]]
                )
                self.assertEqual(calculated_maximum, maximum_value)
            else:
                self.assertEqual(threshold_index, input_index)
                self.assertEqual(threshold_name, threshold_names[input_index])
                self.assertEqual(maximum_value, thresholds[threshold_name])
                self.assertLessEqual(value, maximum_value)

                _, previous_name, previous_maximum = model.get_threshold_for_value(
                    get_test_value(input_index - 1),
                    thresholds
                )
                self.assertGreater(value, previous_maximum)


if __name__ == '__main__':
    unittest.main()
