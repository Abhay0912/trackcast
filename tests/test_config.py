from trackcast.config import RAW_FILES, ROOT, SIGNIFICANT_DELAY_MIN


def test_root_contains_claude_md():
    assert (ROOT / "CLAUDE.md").exists()


def test_raw_files_point_into_data_raw():
    assert all(p.parent.name == "raw" for p in RAW_FILES.values())


def test_hurdle_threshold_matches_proposal():
    assert SIGNIFICANT_DELAY_MIN == 15
