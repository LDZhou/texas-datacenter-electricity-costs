from paper_pipeline.scripts.analyze_results import MODE_LABELS


def test_production_analysis_modes_have_labels():
    production_modes = {"full_tx", "generation_storage"}
    assert production_modes <= MODE_LABELS.keys()
