from polycraft_nov_data.dataloader import polycraft_dataloaders, polycraft_dataset


train_loader, valid_loader, test_loader = polycraft_dataloaders(batch_size=1, shuffle=False)
dataset = polycraft_dataset()


def test_polycraft_len():
    # check length of datasets
    assert len(train_loader) == 5988
    assert len(valid_loader) == 1282
    assert len(test_loader) == 1282
    assert len(dataset) == len(train_loader) + len(valid_loader) + len(test_loader)
