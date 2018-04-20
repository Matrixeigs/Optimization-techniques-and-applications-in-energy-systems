"""
To test the functions of unit commitment problems
"""
def test_assert(cmd):
    try:
        assert cmd in {'ISOLATE_CMD', 'CONNECT_CMD'}
        print(0)
    except AssertionError:
        print(1)

if __name__=="__main__":
    test_assert('CMD')