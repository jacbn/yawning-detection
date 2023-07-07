import unittest
from yawnnlib.preprocessing import eimuModelInput
from yawnnlib.utils import config

import io
import sys

PROJECT_ROOT = config.PROJECT_ROOT
config.set("YAWN_TIME", 2)

class TestModelInput(unittest.TestCase):
    def test_cache(self):
        YAWN_TIME = config.get("YAWN_TIME")
        
        config.set("ENABLE_CACHING", False)
        # redirect stdout to capture print statements
        sys.stdout = captured = io.StringIO() 
        data1, annotations1 = eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=1/32).applyModelTransformOnCachedPath(f"{PROJECT_ROOT}/test/test_data/short1.eimu")
        # check that we did not use the cache
        self.assertFalse("(read from cache)" in captured.getvalue())
        
        config.set("ENABLE_CACHING", True)
        # initialise the cache so we are certain the value is in it
        _ = eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=1/32).applyModelTransformOnCachedPath(f"{PROJECT_ROOT}/test/test_data/short1.eimu")
        
        sys.stdout = captured = io.StringIO()
        data2, annotations2 = eimuModelInput.EimuModelInput(windowSize=YAWN_TIME, windowSep=1/32).applyModelTransformOnCachedPath(f"{PROJECT_ROOT}/test/test_data/short1.eimu")
        # check that we did use the cache
        self.assertTrue("(read from cache)" in captured.getvalue())
        
        # check that the data is the same
        self.assertTrue((data1 == data2).all())
        self.assertTrue((annotations1 == annotations2).all())
        
if __name__ == "__main__": # pragma: no cover
    unittest.main()