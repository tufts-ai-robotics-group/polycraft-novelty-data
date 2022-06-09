from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset


def test_polycraft_len():
    train_loader, valid_loader, test_loader = polycraft_dataloaders(batch_size=1, shuffle=False)
    assert len(train_loader) == 7037
    assert len(valid_loader) == 873
    assert len(test_loader) == 890
    # check length of datasets
    train_loader, valid_loader, test_loader = polycraft_dataloaders(
        batch_size=1, include_novel=True, shuffle=False)
    dataset = polycraft_dataset()
    assert len(train_loader) == 7037
    assert len(valid_loader) == 1205
    assert len(test_loader) == 4420
    assert len(dataset) == len(train_loader) + len(valid_loader) + len(test_loader)
