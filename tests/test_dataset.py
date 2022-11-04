from polycraft_nov_data.dataset import NovelCraft, EpisodeDataset


def test_novelcraft_len():
    # check length of datasets
    train_set = NovelCraft("train")
    assert len(train_set) == 7037
    valid_set = NovelCraft("valid")
    assert len(valid_set) == 1205
    test_set = NovelCraft("test")
    assert len(test_set) == 4420
    valid_novel_set = NovelCraft("valid_novel")
    assert len(valid_novel_set) == 332
    test_novel_set = NovelCraft("test_novel")
    assert len(test_novel_set) == 3530


def test_novelcraft_plus_len():
    # check length of datasets
    train_set = NovelCraft("train", training_plus=True)
    assert len(train_set) == 132673


def test_episode_len():
    # check length of datasets
    train_set = EpisodeDataset("train")
    assert len(train_set) == 3917
    valid_set = EpisodeDataset("valid")
    assert len(valid_set) == 489
    test_set = EpisodeDataset("test")
    assert len(test_set) == 14718
    test_novel_set = EpisodeDataset("test_novel")
    assert len(test_novel_set) == 14227
