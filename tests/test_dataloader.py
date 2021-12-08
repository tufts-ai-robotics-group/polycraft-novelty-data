from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset


def test_polycraft_len():
    train_loader, valid_loader, test_loader = polycraft_dataloaders(batch_size=1, shuffle=False)
    assert len(train_loader) == 24354
    assert len(valid_loader) == 5218
    assert len(test_loader) == 5218
    # check length of datasets
    train_loader, valid_loader, test_loader = polycraft_dataloaders(
        batch_size=1, include_novel=True, shuffle=False)
    dataset = polycraft_dataset()
    assert len(train_loader) == 24354
    assert len(valid_loader) == 11117
    assert len(test_loader) == 11117
    assert len(dataset) == len(train_loader) + len(valid_loader) + len(test_loader)
