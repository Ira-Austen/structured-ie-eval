"""
Unit Tests for Speed Limit and Full Novel Processing Pipelines.
"""

import os
import unittest
from src.runner.run_full_novel import load_novel_chapters, chunk_chapter_text
from src.runner.run_speed_limit import slice_chunks


class TestNovelPipeline(unittest.TestCase):
    def setUp(self):
        self.novel_path = os.path.join(os.path.dirname(__file__), "..", "data", "doupo_full.txt")

    def test_load_novel_chapters(self):
        if not os.path.exists(self.novel_path):
            self.skipTest("doupo_full.txt not found")
        chapters = load_novel_chapters(self.novel_path, max_chapters=10)
        self.assertGreaterEqual(len(chapters), 5)
        self.assertEqual(chapters[1]["chapter_idx"], 1)
        self.assertIn("陨落的天才", chapters[1]["title"])
        self.assertGreater(chapters[1]["char_length"], 1000)

    def test_chunking_logic(self):
        text = "斗之气，三段！少年面无表情，唇角有着一抹自嘲。" * 50
        chunks = chunk_chapter_text(text, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["start"], 0)
        self.assertLessEqual(len(chunks[0]["text"]), 300)

    def test_slice_chunks_speed_limit(self):
        text = "测试长文本分片" * 100
        chunks = slice_chunks(text, chunk_size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["start"], 0)


if __name__ == "__main__":
    unittest.main()
