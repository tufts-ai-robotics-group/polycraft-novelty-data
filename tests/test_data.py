from polycraft_nov_data.data import polycraft_data


train_loader, valid_loader, test_loader = polycraft_data(batch_size=1, shuffle=False)


def test_polycraft_len():
    # check length of datasets
    assert len(train_loader) == 4532
    assert len(valid_loader) == 1510
    assert len(test_loader) == 1510
