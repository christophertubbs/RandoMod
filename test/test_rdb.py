import unittest

from randomod.data.rdb import RDB

import pathlib
import io

TEST_ROOT_DIRECTORY = pathlib.Path(__file__).parent
RESOURCE_DIRECTORY = TEST_ROOT_DIRECTORY / "resources"

TEST_TIMESERIES_PATH = RESOURCE_DIRECTORY / "alabama.rdb"
TEST_STATISTICS_PATH = RESOURCE_DIRECTORY / "statistics.rdb"


class RDBTesting(unittest.TestCase):
    def test_reading(self):
        rdb_text = TEST_TIMESERIES_PATH.read_text()
        timeseries = RDB.parse(rdb_text)

        self.assertEqual(len(timeseries), 229)  # add assertion here

        for site_code, time_series in timeseries.items():
            try:
                frame = time_series.to_frame()
                del frame
            except BaseException as exception:
                self.fail(f"Could not create a dataframe for {site_code}")

        rdb_text = TEST_STATISTICS_PATH.read_text()
        statistics = RDB.parse(rdb_text)

        for site_code, statistics_data in statistics.items():
            try:
                frame = statistics_data.to_frame()
                del frame
            except BaseException as exception:
                self.fail(f"Could not create a data frame for the statistics at: {site_code}")


if __name__ == '__main__':
    unittest.main()
