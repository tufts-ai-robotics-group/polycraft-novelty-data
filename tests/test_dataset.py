from polycraft_nov_data.dataset import NovelCraft, EpisodeDataset


def test_novelcraft_len():
    train_set = NovelCraft("train")
    valid_set = NovelCraft("valid")
    test_set = NovelCraft("test")
    valid_novel_set = NovelCraft("valid_novel")
    test_novel_set = NovelCraft("test_novel")
    # check length of datasets
    assert len(train_set) == 7037
    assert len(valid_set) == 1205
    assert len(test_set) == 4420
    assert len(valid_novel_set) == 332
    assert len(test_novel_set) == 3530


def test_novelcraft_plus_len():
    train_set = NovelCraft("train", training_plus=True)
    # check length of datasets
    assert len(train_set) == 138468


def test_episode_len():
    train_set = EpisodeDataset("train")
    valid_set = EpisodeDataset("valid")
    test_set = EpisodeDataset("test")
    test_novel_set = EpisodeDataset("test_novel")
    # check length of datasets
    assert len(train_set) == 3917
    assert len(test_set) == 14718
    assert len(valid_set) == 489
    assert len(test_novel_set) == 14227
