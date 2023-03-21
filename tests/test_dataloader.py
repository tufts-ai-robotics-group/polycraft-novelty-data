from polycraft_nov_data.dataloader import novelcraft_dataloader, novelcraft_plus_dataloader, \
    episode_dataloader


def test_novelcraft_loader_len():
    train_loader = novelcraft_dataloader("train", batch_size=1)
    assert len(train_loader) == 7037
    valid_loader = novelcraft_dataloader("valid", batch_size=1)
    assert len(valid_loader) == 1205
    test_loader = novelcraft_dataloader("test", batch_size=1)
    assert len(test_loader) == 4420


def test_novelcraft_plus_loader_len():
    train_loader = novelcraft_plus_dataloader("train", batch_size=1)
    assert len(train_loader) == 132673


def test_episode_loader_len():
    train_loader = episode_dataloader("train", batch_size=1)
    assert len(train_loader) == 3917
    valid_loader = episode_dataloader("valid", batch_size=1)
    assert len(valid_loader) == 489
    test_loader = episode_dataloader("test", batch_size=1)
    assert len(test_loader) == 26608
    test_novel_loader = episode_dataloader("test_novel", batch_size=1)
    assert len(test_novel_loader) == 26117
