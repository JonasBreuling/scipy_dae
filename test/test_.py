import pytest

def test_dummy1():
    assert 1 == 1
    assert 2 > 1

def test_dummy2():
    assert True

if __name__ == "__main__":
    test_dummy1()
    test_dummy2()