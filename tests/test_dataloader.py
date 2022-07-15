from polycraft_nov_data.dataloader import novelcraft_dataloader, episode_dataloader


def test_novelcraft_loader_len():
    train_loader = novelcraft_dataloader("train", batch_size=1)
    assert len(train_loader) == 7037
    valid_loader = novelcraft_dataloader("valid", batch_size=1)
    assert len(valid_loader) == 1205
    test_loader = novelcraft_dataloader("test", batch_size=1)
    assert len(test_loader) == 4420


def test_episode_loader_len():
    train_loader = episode_dataloader("train", batch_size=1)
    assert len(train_loader) == 4897
    test_loader = episode_dataloader("test", batch_size=1)
    assert len(test_loader) == 14227
